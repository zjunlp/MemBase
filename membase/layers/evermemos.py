from .base import MemBaseLayer 
from ..baselines.evermemos.online_memory.config import OnlineMemoryManagerConfig
from ..baselines.evermemos.online_memory.memory_manager import OnlineMemoryManager 
from ..configs.evermemos import EverMemOSConfig
from ..utils import (
    PatchSpec,
    make_attr_patch,
    token_monitor,
)
from ..model_types.memory import MemoryEntry
from ..model_types.dataset import Message
import json
import os
from typing import Any, ClassVar


class EverMemOSLayer(MemBaseLayer):
    
    layer_type: ClassVar[str] = "EverMemOS"
    
    def __init__(self, config: EverMemOSConfig) -> None:
        """Create an interface of EverMemOS. The implementation is based on the 
        [official implementation](https://github.com/EverMind-AI/EverMemOS)."""
        internal_config = OnlineMemoryManagerConfig(
            group_id=config.user_id,
            llm=config.llm_config,
            embedding=config.embedding_config,
            boundary=config.boundary_config,
            clustering=config.clustering_config,
            profile=config.profile_config,
            retrieval=config.retrieval_config,
            extraction=config.extraction_config,
        )
        self.memory_layer = OnlineMemoryManager(internal_config)
        self.config = config 
    
    def add_message(self, message: Message, **kwargs: Any) -> None:
        message_dict = {
            "role": message.role,
            "name": message.name,
            "content": message.content,
        }
        self.memory_layer.add_message(
            message_dict, 
            timestamp=message.timestamp,
            **kwargs,
        )
    
    def add_messages(self, messages: list[Message], **kwargs: Any) -> None:
        for message in messages:
            self.add_message(message, **kwargs)
    
    def retrieve(
        self,
        query: str,
        k: int = 10,
        **kwargs: Any,
    ) -> list[MemoryEntry]:
        results = self.memory_layer.retrieve(query, k=k, **kwargs)
        
        # Format results. 
        outputs = []  
        for doc, score in results:
            # See https://github.com/EverMind-AI/EverMemOS/blob/v1.1.0/evaluation/src/adapters/evermemos_adapter.py#L565.  
            subject = doc.get("subject", "N/A")
            episode = doc.get("episode", "N/A")
            doc_text = f"{subject}: {episode}"
            memory_entry = MemoryEntry(
                content=doc.get("episode", ""),
                metadata={
                    "subject": doc.get("subject", ""),
                    "summary": doc.get("summary", ""),
                    "score": float(score), 
                },
                used_content=doc_text,
            )
            outputs.append(memory_entry)
        return outputs
    
    def delete(self, memory_id: str) -> bool:
        raise NotImplementedError(
            "EverMemOS (online version) does not support deleting existing memories."
        )
    
    def update(self, memory_id: str, **kwargs: Any) -> bool:
        raise NotImplementedError(
            "EverMemOS (online version) does not support updating existing memories."
        )
    
    def save_memory(self) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Save config. 
        config_path = os.path.join(self.config.save_dir, "config.json")
        config_dict = {
            "layer_type": self.layer_type,
            **self.config.model_dump(mode="python"),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(
                config_dict, 
                f, 
                indent=4
            )
        
        # Save manager state. 
        self.memory_layer.save(self.config.save_dir)
    
    def load_memory(self, user_id: str | None = None) -> bool:
        if user_id is None:
            user_id = self.config.user_id
        
        config_path = os.path.join(self.config.save_dir, "config.json")
        if not os.path.exists(config_path):
            return False
        
        # Load and update config.
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        if user_id != config_dict["user_id"]:
            raise ValueError(
                f"The user id in the config file ({config_dict['user_id']}) "
                f"does not match the user id ({user_id}) in the function call."
            )
        
        config = EverMemOSConfig(**config_dict)
        internal_config = OnlineMemoryManagerConfig(
            group_id=config.user_id,
            llm=config.llm_config,
            embedding=config.embedding_config,
            boundary=config.boundary_config,
            clustering=config.clustering_config,
            profile=config.profile_config,
            retrieval=config.retrieval_config,
            extraction=config.extraction_config,
        )
        self.memory_layer = OnlineMemoryManager(internal_config)

        is_success = self.memory_layer.load(self.config.save_dir)
        if not is_success:
            return False
        self.config = config
        return True
    
    def flush(self) -> None:
        self.memory_layer.flush()

    def get_patch_specs(self) -> list[PatchSpec]:
        getter, setter = make_attr_patch(self.memory_layer.llm_model, "generate")
        # In EverMemOS, update operations occur in both cluster status updates and profile updates.
        # Cluster updates are based on the encoder, while profile updates require usage of an LLM. 
        spec = PatchSpec(
            name = f"{self.memory_layer.llm_model.__class__.__name__}.generate",
            getter=getter,
            setter=setter,
            wrapper=token_monitor(
                extract_model_name=lambda *args, **kwargs: (
                    self.config.llm_config.model, {}
                ),
                extract_input_dict=lambda *args, **kwargs: {
                    "messages": kwargs.get("prompt", args[0] if len(args) > 0 else ""),
                    "metadata": {
                        "op_type": (
                            "update" if 
                                "You are a personal profile extraction expert" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are a person profile project experiences" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "Please analyze the latest user-AI conversation" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are an evidence completion assistant" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are a group content analysis expert" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are a group behavior analysis expert" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are a group profile aggregation expert" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "You are a personal profile analysis expert" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "你是一位个人档案提取专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "你是一位个人档案项目经验提取专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "请分析以下最新的用户-AI对话" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "你是一个证据完成助手" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "你是一位群组内容分析专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                ) 
                                or "你是一位群组行为分析专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                )
                                or "你是一位群组档案聚合专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                )
                                or "你是一位个人档案分析专家" in kwargs.get(
                                    "prompt", 
                                    args[0] if len(args) > 0 else ""
                                )
                            else "generation"
                        ) 
                    }
                },
                extract_output_dict=lambda result: {
                    "messages": result
                },
            ),
        ) 
        return [spec]
