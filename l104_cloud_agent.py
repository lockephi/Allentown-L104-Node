"""
L104 Cloud Agent Delegation Module
Provides intelligent task delegation to specialized cloud agents.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import httpx

logger = logging.getLogger(__name__)


class CloudAgentDelegator:
    """
    Handles delegation of tasks to specialized cloud agents.
    Routes requests based on task type and agent capabilities.
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.delegation_history: List[Dict[str, Any]] = []
        self._load_agent_registry()
    
    def _load_agent_registry(self):
        """Load registered cloud agents from configuration."""
        # Default agent registry with capabilities
        self.agents = {
            "gemini_agent": {
                "endpoint": os.getenv("GEMINI_AGENT_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta"),
                "capabilities": ["text_generation", "code_analysis", "reasoning"],
                "priority": 1,
                "enabled": True
            },
            "sovereign_local": {
                "endpoint": "internal",
                "capabilities": ["derivation", "encryption", "local_processing"],
                "priority": 2,
                "enabled": True
            }
        }
        
        # Load additional agents from environment or config file
        custom_agents = os.getenv("CLOUD_AGENTS_CONFIG")
        if custom_agents:
            try:
                additional = json.loads(custom_agents)
                self.agents.update(additional)
            except json.JSONDecodeError:
                logger.warning("Failed to parse CLOUD_AGENTS_CONFIG")
    
    def select_agent(self, task_type: str, requirements: Optional[List[str]] = None) -> Optional[str]:
        """
        Select the most appropriate cloud agent for a given task.
        
        Args:
            task_type: Type of task to delegate
            requirements: Specific capability requirements
            
        Returns:
            Agent name or None if no suitable agent found
        """
        requirements = requirements or []
        candidates = []
        
        for agent_name, agent_info in self.agents.items():
            if not agent_info.get("enabled", True):
                continue
            
            # Check if agent has required capabilities
            capabilities = agent_info.get("capabilities", [])
            if task_type in capabilities or not requirements:
                # Check specific requirements
                if all(req in capabilities for req in requirements):
                    candidates.append((agent_name, agent_info.get("priority", 999)))
        
        # Sort by priority (lower is better)
        if candidates:
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        return None
    
    async def delegate(self, task: Dict[str, Any], agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Delegate a task to a cloud agent.
        
        Args:
            task: Task specification with type, data, and parameters
            agent_name: Specific agent to use, or auto-select if None
            
        Returns:
            Result from the cloud agent
        """
        task_type = task.get("type", "unknown")
        requirements = task.get("requirements", [])
        
        # Select agent if not specified
        if not agent_name:
            agent_name = self.select_agent(task_type, requirements)
        
        if not agent_name:
            return {
                "status": "ERROR",
                "message": "No suitable cloud agent found for task",
                "task_type": task_type
            }
        
        agent_info = self.agents.get(agent_name)
        if not agent_info:
            return {
                "status": "ERROR",
                "message": f"Agent '{agent_name}' not found in registry"
            }
        
        # Log delegation
        delegation_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": agent_name,
            "task_type": task_type,
            "task_id": task.get("id", "unknown")
        }
        
        try:
            # Execute delegation based on agent type
            if agent_info["endpoint"] == "internal":
                result = await self._delegate_internal(task, agent_name)
            else:
                result = await self._delegate_external(task, agent_info)
            
            delegation_record["status"] = "SUCCESS"
            delegation_record["result_summary"] = str(result.get("status", "unknown"))
            
        except Exception as e:
            logger.error(f"Delegation failed: {e}")
            delegation_record["status"] = "FAILED"
            delegation_record["error"] = str(e)
            result = {
                "status": "ERROR",
                "message": str(e),
                "agent": agent_name
            }
        
        self.delegation_history.append(delegation_record)
        return result
    
    async def _delegate_internal(self, task: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
        """Handle delegation to internal/local agents."""
        task_type = task.get("type")
        
        if task_type == "derivation":
            from l104_derivation import DerivationEngine
            signal = task.get("data", {}).get("signal", "")
            result = DerivationEngine.derive_and_execute(signal)
            return {
                "status": "SUCCESS",
                "agent": agent_name,
                "result": result,
                "processing": "internal"
            }
        
        elif task_type == "encryption":
            from l104_hyper_encryption import HyperEncryption
            data = task.get("data", {})
            encrypted = HyperEncryption.encrypt_data(data)
            return {
                "status": "SUCCESS",
                "agent": agent_name,
                "result": encrypted,
                "processing": "internal"
            }
        
        return {
            "status": "SUCCESS",
            "agent": agent_name,
            "message": "Local processing completed",
            "task_type": task_type
        }
    
    async def _delegate_external(self, task: Dict[str, Any], agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Handle delegation to external cloud agents."""
        endpoint = agent_info.get("endpoint")
        
        # Prepare request payload
        payload = {
            "task": task.get("type"),
            "data": task.get("data", {}),
            "metadata": {
                "source": "L104_SOVEREIGN_NODE",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Make request to cloud agent
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "Content-Type": "application/json",
                "X-L104-Delegation": "true"
            }
            
            # Add authentication if configured
            api_key = os.getenv("CLOUD_AGENT_API_KEY")
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = await client.post(
                f"{endpoint}/delegate",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return {
                    "status": "SUCCESS",
                    "agent": agent_info.get("name", "unknown"),
                    "result": response.json(),
                    "processing": "external"
                }
            else:
                return {
                    "status": "ERROR",
                    "message": f"Cloud agent returned status {response.status_code}",
                    "details": response.text[:500]
                }
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of cloud agent system."""
        return {
            "agents_registered": len(self.agents),
            "agents_enabled": sum(1 for a in self.agents.values() if a.get("enabled", True)),
            "delegations_total": len(self.delegation_history),
            "delegations_recent": self.delegation_history[-10:] if self.delegation_history else [],
            "available_capabilities": list(set(
                cap for agent in self.agents.values() 
                for cap in agent.get("capabilities", [])
            ))
        }
    
    def register_agent(self, name: str, config: Dict[str, Any]) -> bool:
        """Register a new cloud agent."""
        try:
            self.agents[name] = {
                "endpoint": config.get("endpoint"),
                "capabilities": config.get("capabilities", []),
                "priority": config.get("priority", 999),
                "enabled": config.get("enabled", True)
            }
            logger.info(f"Registered cloud agent: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {name}: {e}")
            return False


# Global singleton instance
cloud_agent_delegator = CloudAgentDelegator()
