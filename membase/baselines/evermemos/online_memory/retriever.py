import asyncio
import json
import time
from online_memory.config import OnlineRetrieverConfig
from online_memory.index_manager import InMemoryIndexManager
from online_memory.prompts import (
    SUFFICIENCY_CHECK_PROMPT, 
    MULTI_QUERY_GENERATION_PROMPT, 
    REFINED_QUERY_PROMPT, 
) 
from typing import (
    Any, 
    Dict, 
    List, 
    Optional, 
    Tuple,
)


class OnlineRetriever:
    """Online retriever with multiple retrieval modes."""
    
    def __init__(
        self,
        index_manager: InMemoryIndexManager,
        config: Optional[OnlineRetrieverConfig] = None,
        llm_provider: Optional[Any] = None,
    ) -> None:
        """
        Initialize the retriever.
        
        Parameters
        ----------
        index_manager : InMemoryIndexManager
            The index manager for BM25 and embedding search.
        config : OnlineRetrieverConfig, optional
            Retriever configuration.
        llm_provider : Any, optional
            LLM provider for agentic retrieval.
        """
        self.index_manager = index_manager
        self.config = config or OnlineRetrieverConfig()
        self.llm_provider = llm_provider
        
        # Initialize reranker service
        self._init_reranker_service()
    
    def _init_reranker_service(self) -> None:
        """Initialize the reranker service from OnlineRetrieverConfig."""
        if self.config.use_reranker:
            from agentic_layer.rerank_service import (
                HybridRerankConfig,
                HybridRerankService,
            )
            
            # Create `HybridRerankConfig` from input parameters.
            rerank_config = HybridRerankConfig(
                primary_provider=self.config.reranker_provider,
                primary_api_key=self.config.reranker_api_key,
                primary_base_url=self.config.reranker_base_url,
                model=self.config.reranker_model,
            )
            self._reranker = HybridRerankService(config=rerank_config)
            print(f"  [Retriever] Initialized HybridRerankService | provider={rerank_config.primary_provider} | model={rerank_config.model}")
        else:
            self._reranker = None
    
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        **kwargs,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve relevant documents for a query.
        
        Parameters
        ----------
        query : str
            Search query.
        k : int, optional
            Maximum number of results to return.
        **kwargs
            Additional parameters such as the mode of retrieval.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            List of (document, score) tuples.
        """
        if not query or len(self.index_manager) == 0:
            return []
        
        effective_k = k if k is not None else self.config.final_top_k
        mode = kwargs.get("retrieval_mode", self.config.retrieval_mode)
        
        if mode == "agentic":
            if self.llm_provider is None:
                print("Warning: Agentic mode requires LLM provider. Falling back to hybrid.")
                return await self._search_hybrid(query, effective_k)
            return await self._search_agentic(query, effective_k)
        elif mode == "bm25_only":
            return self._search_bm25_only(query, effective_k)
        elif mode == "emb_only":
            return await self._search_emb_only(query, effective_k)
        else:  
            return await self._search_hybrid(query, effective_k)
    
    def _search_bm25_only(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """BM25-only search."""
        results = self.index_manager.search_bm25(query, top_n=k)
        return results
    
    async def _search_emb_only(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Embedding-only search."""
        results = await self.index_manager.search_embedding(query, top_n=k)
        return results
    
    async def _search_hybrid(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Hybrid search with RRF fusion.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L491. 
        """
        # Adjust `bm25_top_n` and `emb_top_n` if `k` is larger
        bm25_n = max(self.config.bm25_top_n, k)
        emb_n = max(self.config.emb_top_n, k)
        
        # Run BM25 and embedding search in parallel
        bm25_task = asyncio.to_thread(
            self.index_manager.search_bm25,
            query,
            bm25_n,
        )
        emb_task = self.index_manager.search_embedding(
            query,
            emb_n,
        )
        
        bm25_results, emb_results = await asyncio.gather(bm25_task, emb_task)
        
        # Handle edge cases
        if not bm25_results and not emb_results:
            print(f"Warning: Both BM25 and embedding search returned no results for query: {query}")
            return []
        elif not bm25_results:
            print(f"Warning: BM25 search returned no results for query: {query}")
            return emb_results[:k]
        elif not emb_results:
            print(f"Warning: Embedding search returned no results for query: {query}")
            return bm25_results[:k]
        
        # RRF fusion
        fused = self._rrf_fusion(bm25_results, emb_results)
        
        print(f"Hybrid search: Emb={len(emb_results)}, BM25={len(bm25_results)}, Fused={len(fused)}, Returning top-{k}")
        return fused[:k]
    
    async def _search_agentic(
        self,
        query: str,
        k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Agentic multi-round LLM-guided retrieval.
        
        Process:
        1. Round 1: Hybrid search -> Top N -> Rerank -> Sufficiency check
        2. If sufficient: return reranked results
        3. If insufficient:
           - Generate refined queries
           - Round 2: Retrieve and merge
           - Rerank combined results -> return final
        
        Parameters
        ----------
        query : str
            Search query.
        k : int
            Maximum number of results to return.
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Agentic Retrieval: {query[:60]}...")
        print(f"{'='*60}")
        print(f"  [Start] Time: {time.strftime('%H:%M:%S')}")
        
        # Determine retrieval sizes based on k
        # Round 1 retrieves k documents for candidate pool
        round1_size = k 
        sufficiency_check_size = min(self.config.sufficiency_check_docs, k)
        
        # Round 1: Hybrid search
        print(f"  [Round 1] Hybrid search for Top {round1_size}...")
        round1_results = await self._search_hybrid(query, round1_size)
        
        if not round1_results:
            print(f"  [Warning] No results from Round 1")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            return []
        
        print(f"  [Round 1] Retrieved {len(round1_results)} documents")
        
        # Rerank for sufficiency check
        round1_results_ = round1_results 
        if self.config.use_reranker:
            print(f"  [Rerank] Reranking to get Top {sufficiency_check_size} for sufficiency check...")
            round1_results_ = await self._rerank_results(query, round1_results, top_n=len(round1_results))
            reranked_for_check = round1_results_[:sufficiency_check_size]
            print(f"  [Rerank] Got {len(reranked_for_check)} documents for sufficiency check")
        else:
            reranked_for_check = round1_results[:sufficiency_check_size]
            print(f"  [No Rerank] Using original Top {sufficiency_check_size} for sufficiency check")
        
        if not reranked_for_check:
            print(f"  [Warning] Reranking failed, falling back to original results")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            return round1_results[:k]
        round1_results = round1_results_
        
        # Sufficiency check
        print(f"  [LLM] Checking sufficiency on Top {len(reranked_for_check)}...")
        is_sufficient, reasoning, missing_info, key_info = await self._check_sufficiency(
            query, reranked_for_check
        )
        
        print(f"  [LLM] Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
        print(f"  [LLM] Reasoning: {reasoning}")
        if key_info:
            print(f"  [LLM] Key Info Found: {', '.join(key_info)}")
        
        if is_sufficient:
            # If sufficient, rerank full round1 results to get final k documents
            final_results = round1_results[:k]
            
            print(f"  [Complete] Sufficient! Final: {len(final_results)} docs")
            print(f"Time taken: {time.time() - start_time:.2f} seconds")
            return final_results
        
        # Round 2: Generate refined queries and search
        print(f"  [Round 2] Insufficient, generating refined queries...")
        print(f"  [Missing] {', '.join(missing_info) if missing_info else 'N/A'}")
        
        if self.config.use_multi_query:
            refined_queries, _ = await self._generate_multi_queries(
                query, reranked_for_check, missing_info, key_info
            )
            print(f"  [Round 2] Generated {len(refined_queries)} queries")
        else:
            refined_query = await self._generate_refined_query(
                query, reranked_for_check, missing_info
            )
            refined_queries = [refined_query]
            print(f"  [Round 2] Generated refined query: {refined_query}...")
        
        # Execute refined queries
        # Each query retrieves k candidates
        round2_retrieval_size = k
        all_round2_results = []
        for i, rq in enumerate(refined_queries, 1):
            print(f"  [Round 2] Searching query {i}: {rq}...")
            r2_results = await self._search_hybrid(rq, round2_retrieval_size)
            all_round2_results.append(r2_results)
            print(f"    Query {i}: Retrieved {len(r2_results)} documents")
        
        # Multi-query RRF fusion
        print(f"  [Multi-RRF] Fusing results from {len(refined_queries)} queries...")
        if len(all_round2_results) > 1:
            round2_results = self._multi_rrf_fusion(all_round2_results)
        elif all_round2_results:
            round2_results = all_round2_results[0]
        else:
            round2_results = []
        
        # Merge Round 1 and Round 2 results
        # Target merge size: 2x k for reranking
        merge_target = k * 2
        
        print(f"  [Merge] Combining Round 1 and Round 2 to {merge_target} documents...")
        round1_ids = {doc.get("event_id", id(doc)) for doc, _ in round1_results}
        round2_unique = [
            (doc, score) for doc, score in round2_results
            if doc.get("event_id", id(doc)) not in round1_ids
        ]
        
        combined = round1_results.copy()
        needed_from_round2 = merge_target - len(combined)
        combined.extend(round2_unique[:needed_from_round2])
        
        duplicates_removed = len(round2_results) - len(round2_unique)
        round2_added = len(round2_unique[:needed_from_round2])
        
        print(
            f"  [Merge] Round1: {len(round1_results)}, Round2 unique added: {round2_added}, duplicates removed: {duplicates_removed}"
        )
        print(f"  [Merge] Combined total: {len(combined)} documents")
        
        # Final rerank
        if self.config.use_reranker and len(combined) > 0:
            print(f"  [Rerank] Reranking {len(combined)} combined documents to get Top {k}...")
            final_results = await self._rerank_results(query, combined, top_n=k)
            print(f"  [Rerank] Final Top {len(final_results)} selected")
        else:
            final_results = combined[:k]
            print(f"  [No Rerank] Returning Top {k} from combined results")
        
        print(f"  [Complete] Final: {len(final_results)} docs")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")
        print(f"{'='*60}\n")
        
        return final_results
    
    async def _check_sufficiency(
        self,
        query: str,
        results: List[Tuple[Dict[str, Any], float]],
    ) -> Tuple[bool, str, List[str], List[str]]:
        """Check if retrieval results are sufficient.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L172. 
        """
        try:
            # Format documents
            docs_text = self._format_documents(results)
            
            prompt = SUFFICIENCY_CHECK_PROMPT.format(
                query=query,
                retrieved_docs=docs_text,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=500,
            )
            
            # Parse JSON response
            result = self._parse_json_response(response)
            
            return (
                result["is_sufficient"],
                result["reasoning"],
                result.get("missing_information", []),
                result.get("key_information_found", [])
            )

        except asyncio.TimeoutError:
            print(f"  ❌ Sufficiency check timeout (30s)")
            # Timeout fallback: assume sufficient
            return True, "Timeout: LLM took too long", [], []
        except Exception as e:
            print(f"  ❌ Sufficiency check failed: {e}")
            import traceback
            traceback.print_exc()
            # Conservative fallback: assume sufficient
            return True, f"Error: {str(e)}", [], []

    async def _generate_refined_query(
        self,
        original_query: str,
        results: List[Tuple[Dict[str, Any], float]],
        missing_info: List[str],
    ) -> str:
        """Generate a refined query using self.llm_provider."""
        try:
            docs_text = self._format_documents(results)
            missing_str = ", ".join(missing_info) if missing_info else "N/A"
            
            prompt = REFINED_QUERY_PROMPT.format(
                original_query=original_query,
                retrieved_docs=docs_text,
                missing_info=missing_str,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=150,
            )
            
            refined_query = self._parse_refined_query(response, original_query)
            
            return refined_query
        
        except asyncio.TimeoutError:
            print(f"  ❌ Query refinement timeout (30s)")
            # Timeout fallback: use original query
            return original_query
        except Exception as e:
            print(f"  ❌ Query refinement failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to original query
            return original_query
    
    async def _generate_multi_queries(
        self,
        original_query: str,
        results: List[Tuple[Dict[str, Any], float]],
        missing_info: List[str],
        key_info: List[str],
    ) -> Tuple[List[str], str]:
        """Generate multiple complementary queries (2-3 queries as per prompt).
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L355. 
        """
        try:
            docs_text = self._format_documents(results)
            missing_str = ", ".join(missing_info) if missing_info else "N/A"
            key_str = ", ".join(key_info) if key_info else "N/A"
            
            prompt = MULTI_QUERY_GENERATION_PROMPT.format(
                original_query=original_query,
                retrieved_docs=docs_text,
                missing_info=missing_str,
                key_info=key_str,
            )
            
            response = await self.llm_provider.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=300,
            )
            
            queries, reasoning = self._parse_multi_query_response(response, original_query)
            
            print(f"  [Multi-Query] Generated {len(queries)} queries:")
            for i, q in enumerate(queries, 1):
                print(f"    Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
            print(f"  [Multi-Query] Strategy: {reasoning}")
            
            return queries, reasoning
        
        except asyncio.TimeoutError:
            print(f"  ❌ Multi-query generation timeout (30s)")
            return [original_query], "Timeout: used original query"
        except Exception as e:
            print(f"  ❌ Multi-query generation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to original query
            return [original_query], f"Error: {str(e)}"
    
    def _format_documents(
        self,
        results: List[Tuple[Dict[str, Any], float]],
    ) -> str:
        """Format documents for LLM consumption.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L21. 
        """
        formatted = []
        
        for i, (doc, _) in enumerate(results, start=1):
            subject = doc.get("subject", "N/A")
            episode = doc.get("episode", "N/A")
            
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            formatted.append(
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
        
        return "\n".join(formatted)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L95. 
        """
        try:
            # Extract JSON (LLM may add extra text before/after)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "is_sufficient" not in result:
                raise ValueError("Missing 'is_sufficient' field")
            
            # Set default values
            result.setdefault("reasoning", "No reasoning provided")
            result.setdefault("missing_information", [])
            result.setdefault("key_information_found", [])
            
            return result
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Failed to parse LLM response: {e}")
            print(f"  Raw response: {response[:200]}...")
            
            # Conservative fallback: assume sufficient to avoid unnecessary second round
            return {
                "is_sufficient": True,
                "reasoning": f"Failed to parse: {str(e)}",
                "missing_information": [],
                "key_information_found": []
            }
    
    def _parse_multi_query_response(self, response: str, original_query: str) -> Tuple[List[str], str]:
        """Parse multi-query generation JSON response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L299. 
        """
        try:
            # Extract JSON
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON object found in response")
            
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
            
            # Validate required fields
            if "queries" not in result or not isinstance(result["queries"], list):
                raise ValueError("Missing or invalid 'queries' field")
            
            queries = result["queries"]
            reasoning = result.get("reasoning", "No reasoning provided")
            
            # Filter and validate queries
            valid_queries = []
            for q in queries:
                if isinstance(q, str) and 5 <= len(q) <= 300:
                    # Avoid identical to original query
                    if q.lower().strip() != original_query.lower().strip():
                        valid_queries.append(q.strip())
            
            # Return at least 1 query
            if not valid_queries:
                print(f"  ⚠️  No valid queries generated, using original")
                return [original_query], "Fallback: used original query"
            
            # Limit to maximum 3 queries
            valid_queries = valid_queries[:3]
            
            print(f"  ✅ Generated {len(valid_queries)} valid queries")
            return valid_queries, reasoning
        
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Failed to parse multi-query response: {e}")
            print(f"  Raw response: {response[:200]}...")
            
            # Fallback: return original query
            return [original_query], f"Parse error: {str(e)}"

    def _parse_refined_query(self, response: str, original_query: str) -> str:
        """
        Parse refined query from LLM response.
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/tools/agentic_utils.py#L140.
        """
        refined = response.strip()
        
        # Remove common prefixes
        prefixes = ["Refined Query:", "Output:", "Answer:", "Query:"]
        for prefix in prefixes:
            if refined.startswith(prefix):
                refined = refined[len(prefix):].strip()
        
        # Validate length
        if len(refined) < 5 or len(refined) > 300:
            print(f"  ⚠️  Invalid refined query length ({len(refined)}), using original")
            return original_query
        
        # Avoid identical query
        if refined.lower() == original_query.lower():
            print(f"  ⚠️  Refined query identical to original, using original")
            return original_query
        
        return refined
    
    def _rrf_fusion(
        self,
        results1: List[Tuple[Dict[str, Any], float]],
        results2: List[Tuple[Dict[str, Any], float]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Fuse two result lists using Reciprocal Rank Fusion (RRF).
        
        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L191. 
        """
        k = self.config.rrf_k
        
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        # Process first result list
        for rank, (doc, _) in enumerate(results1, start=1):
            doc_id = doc.get("event_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        # Process second result list
        for rank, (doc, _) in enumerate(results2, start=1):
            doc_id = doc.get("event_id", id(doc))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    def _multi_rrf_fusion(
        self,
        results_list: List[List[Tuple[Dict[str, Any], float]]],
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Fuse multiple result lists using RRF."""
        if not results_list:
            return []
        
        if len(results_list) == 1:
            return results_list[0]
        
        k = self.config.rrf_k
        doc_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}
        
        for results in results_list:
            for rank, (doc, _) in enumerate(results, start=1):
                doc_id = doc.get("event_id", id(doc))
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [(doc_map[doc_id], score) for doc_id, score in sorted_docs]
    
    async def multi_query_retrieve(
        self,
        queries: List[str],
        k: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve using multiple queries and fuse results.
        
        Parameters
        ----------
        queries : List[str]
            List of queries.
        k : int, optional
            Maximum number of results to return.
            If not provided, uses config.final_top_k.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            Fused results from all queries.
        """
        if not queries:
            return []
        
        effective_k = k if k is not None else self.config.final_top_k
        
        # Run all queries in parallel, each retrieves effective_k candidates
        tasks = [self.retrieve(q, k=effective_k) for q in queries]
        all_results = await asyncio.gather(*tasks)
        
        # Multi-query RRF fusion
        return self._multi_rrf_fusion(all_results)[:effective_k]
    
    # ========================
    # Reranker Methods
    # ========================
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Tuple[Dict[str, Any], float]],
        top_n: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank retrieval results using neural reranker.
        
        For documents containing event_log:
        - Format as multi-line text: time + each atomic_fact on separate line

        See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos/stage3_memory_retrivel.py#L869. 
        
        Features:
        - Batch processing with configurable batch size
        - Retry with exponential backoff
        - Timeout protection per batch
        - Fallback to original ranking when success rate is too low
        - Controlled concurrent batch processing
        
        Parameters
        ----------
        query : str
            The query to rerank against.
        results : List[Tuple[Dict, float]]
            Initial retrieval results.
        top_n : int, optional
            Number of documents to return after reranking.
            Defaults to config.final_top_k.
        
        Returns
        -------
        List[Tuple[Dict, float]]
            Reranked results with reranker scores.
        """
        if not results:
            return []
        
        # Default to `final_top_k` if `top_n` is not specified
        effective_top_n = top_n if top_n is not None else self.config.final_top_k
        
        batch_size = self.config.reranker_batch_size
        max_retries = self.config.reranker_max_retries
        retry_delay = self.config.reranker_retry_delay
        timeout = self.config.reranker_timeout
        fallback_threshold = self.config.reranker_fallback_threshold
        max_concurrent = self.config.reranker_concurrent_batches
        reranker_instruction = self.config.reranker_instruction
        
        # Step 1: Format documents for reranker
        docs_with_text = []
        doc_texts = []
        original_indices = []
        
        for idx, (doc, _) in enumerate(results):
            # Prefer event_log to format text (if exists)
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                event_log = doc["event_log"]
                time_str = event_log.get("time", "")
                atomic_facts = event_log.get("atomic_fact", [])
                
                if isinstance(atomic_facts, list) and atomic_facts:
                    formatted_lines = []
                    if time_str:
                        formatted_lines.append(time_str)
                    
                    for fact in atomic_facts:
                        if isinstance(fact, dict) and "fact" in fact:
                            formatted_lines.append(fact["fact"])
                        elif isinstance(fact, str):
                            formatted_lines.append(fact)
                    
                    formatted_text = "\n".join(formatted_lines)
                    docs_with_text.append(doc)
                    doc_texts.append(formatted_text)
                    original_indices.append(idx)
                    continue
            
            # Fallback to episode field
            if episode_text := doc.get("episode"):
                docs_with_text.append(doc)
                doc_texts.append(episode_text)
                original_indices.append(idx)
        
        if not doc_texts:
            return results[:effective_top_n]
        
        # Check if reranker is available
        if self._reranker is None:
            print("  [Rerank] Warning: Reranker not initialized, using original ranking")
            return results[:effective_top_n]
        
        print(f"  [Rerank] Reranking {len(doc_texts)} documents in batches of {batch_size}...")
        
        # Step 2: Split into batches
        batches = []
        for i in range(0, len(doc_texts), batch_size):
            batch = doc_texts[i:i + batch_size]
            batches.append((i, batch))
        
        print(f"  [Rerank] Split into {len(batches)} batches")
        
        # Process single batch with retry
        async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
            for attempt in range(max_retries):
                try:
                    batch_results = await asyncio.wait_for(
                        self._reranker.rerank_documents(
                            query, batch_texts, instruction=reranker_instruction
                        ),
                        timeout=timeout
                    )
                    
                    # Adjust indices to global indices
                    for item in batch_results["results"]:
                        item["global_index"] = start_idx + item["index"]
                    
                    if attempt > 0:
                        print(f"    ✓ Batch at {start_idx} succeeded on attempt {attempt + 1}")
                    return batch_results["results"]
                    
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⏱️  Batch at {start_idx} timeout (attempt {attempt + 1}), retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    ❌ Batch at {start_idx} timeout after {max_retries} attempts")
                        return []
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"    ⚠️  Batch at {start_idx} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"    ❌ Batch at {start_idx} failed after {max_retries} attempts: {e}")
                        return []
        
        # Step 3: Process batches with controlled concurrency
        batch_results_list = []
        successful_batches = 0
        
        for group_start in range(0, len(batches), max_concurrent):
            group_batches = batches[group_start:group_start + max_concurrent]
            
            print(f"    Processing batch group {group_start // max_concurrent + 1} ({len(group_batches)} batches in parallel)...")
            
            tasks = [
                process_batch_with_retry(start_idx, batch)
                for start_idx, batch in group_batches
            ]
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in group_results:
                if isinstance(result, list) and result:
                    batch_results_list.append(result)
                    successful_batches += 1
                else:
                    batch_results_list.append([])
            
            # Inter-group delay
            if group_start + max_concurrent < len(batches):
                await asyncio.sleep(0.3)
        
        # Step 4: Merge results and apply fallback strategy
        all_rerank_results = []
        for batch_results in batch_results_list:
            all_rerank_results.extend(batch_results)
        
        success_rate = successful_batches / len(batches) if batches else 0.0
        print(f"  [Rerank] Success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)")
        
        # Fallback: complete failure
        if not all_rerank_results:
            print("  [Rerank] ⚠️ All batches failed, using original ranking")
            return results[:effective_top_n]
        
        # Fallback: success rate too low
        if success_rate < fallback_threshold:
            print(f"  [Rerank] ⚠️ Success rate too low ({success_rate:.1%} < {fallback_threshold:.1%}), using original ranking")
            return results[:effective_top_n]
        
        print(f"  [Rerank] Complete: {len(all_rerank_results)} documents scored")
        
        # Step 5: Sort by reranker score and return top-N
        sorted_results = sorted(
            all_rerank_results,
            key=lambda x: x["score"],
            reverse=True
        )[:effective_top_n]
        
        # Map back to original documents
        final_results = [
            (results[original_indices[item["global_index"]]][0], item["score"])
            for item in sorted_results
        ]
        
        return final_results
