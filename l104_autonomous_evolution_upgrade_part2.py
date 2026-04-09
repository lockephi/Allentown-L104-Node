#!/usr/bin/env python3
"""
L104 Autonomous Evolutionary Process Upgrade - Part 2
Continuous evolution, monitoring, and integration
"""

import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import numpy as np
from l104_autonomous_evolution_upgrade import (
    AutonomousEvolutionEngine, EvolutionaryState, 
    EvolutionarySolution, EvolutionPhase, EvolutionStrategy
)

# ============================================================================
# Continuous Evolution Manager
# ============================================================================

class ContinuousEvolutionManager:
    """Manages continuous, autonomous evolutionary processes"""
    
    def __init__(self, 
                 engine: AutonomousEvolutionEngine,
                 evolution_interval: int = 300,  # 5 minutes
                 monitoring_interval: int = 60,   # 1 minute
                 adaptation_window: int = 24):    # 24 hours
        
        self.engine = engine
        self.evolution_interval = evolution_interval
        self.monitoring_interval = monitoring_interval
        self.adaptation_window = adaptation_window
        
        # Evolution tasks
        self.evolution_task = None
        self.monitoring_task = None
        self.optimization_task = None
        
        # Performance tracking
        self.start_time = datetime.now()
        self.evolution_cycles = 0
        self.improvement_history = []
        self.innovation_history = []
        
        # Integration points
        self.l104_api_url = "http://localhost:8004"
        self.integration_enabled = True
        
        # Evolution goals
        self.target_fitness = None
        self.target_diversity = 0.2
        self.target_innovation_rate = 0.15
        
        print(f"🚀 Initializing Continuous Evolution Manager")
        print(f"   Evolution interval: {evolution_interval}s")
        print(f"   Monitoring interval: {monitoring_interval}s")
        print(f"   Adaptation window: {adaptation_window}h")
    
    async def start_continuous_evolution(self, 
                                       fitness_function: Callable[[Dict[str, Any]], float],
                                       param_ranges: Dict[str, tuple]):
        """Start continuous evolutionary process"""
        print(f"\n🌀 Starting continuous evolution...")
        
        # Initialize population if needed
        if not self.engine.population:
            self.engine.initialize_population(param_ranges, fitness_function)
        
        # Start evolution loop
        self.evolution_task = asyncio.create_task(
            self._evolution_loop(fitness_function)
        )
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop()
        )
        
        # Start optimization
        self.optimization_task = asyncio.create_task(
            self._optimization_loop()
        )
        
        print(f"   ✅ Continuous evolution started")
        print(f"   📊 Evolution engine: {self.engine.name}")
        print(f"   🎯 Target fitness: {self.target_fitness}")
        print(f"   🔄 Evolution cycles will run every {self.evolution_interval}s")
        
        return True
    
    async def _evolution_loop(self, fitness_function: Callable):
        """Main evolution loop"""
        while True:
            try:
                # Run one evolution cycle
                self.engine.evolve_generation(fitness_function)
                self.evolution_cycles += 1
                
                # Check for significant improvement
                if len(self.engine.fitness_history) > 1:
                    improvement = self.engine.fitness_history[-1] - self.engine.fitness_history[-2]
                    self.improvement_history.append(improvement)
                    
                    if improvement > 0.01:  # Significant improvement
                        print(f"   📈 Significant improvement: +{improvement:.4f}")
                        
                        # Archive breakthrough
                        self._archive_breakthrough()
                
                # Integrate with L104 system
                if self.integration_enabled:
                    await self._integrate_with_l104()
                
                # Wait for next evolution cycle
                await asyncio.sleep(self.evolution_interval)
                
            except Exception as e:
                print(f"   ⚠️ Evolution loop error: {e}")
                await asyncio.sleep(10)  # Wait before retry
    
    async def _monitoring_loop(self):
        """Monitor evolutionary process"""
        while True:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                
                # Log metrics
                self._log_metrics(metrics)
                
                # Check for stagnation
                if self._check_stagnation():
                    print(f"   ⚠️ Evolution stagnation detected")
                    await self._trigger_innovation()
                
                # Check for convergence
                if self._check_convergence():
                    print(f"   ⚠️ Premature convergence detected")
                    await self._increase_diversity()
                
                # Update evolution parameters if needed
                if self._should_adjust_parameters():
                    await self._adjust_evolution_parameters()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"   ⚠️ Monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _optimization_loop(self):
        """Optimize evolutionary process parameters"""
        while True:
            try:
                # Analyze performance trends
                trends = self._analyze_performance_trends()
                
                # Optimize based on trends
                if trends.get('fitness_trend', 0) < 0.001:  # Slow improvement
                    print(f"   🔧 Optimizing evolution parameters (slow improvement)")
                    await self._optimize_for_speed()
                
                if trends.get('diversity_trend', 0) < -0.01:  # Decreasing diversity
                    print(f"   🔧 Optimizing evolution parameters (low diversity)")
                    await self._optimize_for_diversity()
                
                if trends.get('innovation_trend', 0) < 0:  # Decreasing innovation
                    print(f"   🔧 Optimizing evolution parameters (low innovation)")
                    await self._optimize_for_innovation()
                
                # Wait for next optimization cycle (every hour)
                await asyncio.sleep(3600)
                
            except Exception as e:
                print(f"   ⚠️ Optimization loop error: {e}")
                await asyncio.sleep(300)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive evolution metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'evolution_cycles': self.evolution_cycles,
            'generation': self.engine.generation,
            'phase': self.engine.state.phase.value,
            
            # Fitness metrics
            'best_fitness': self.engine.population[0].fitness if self.engine.population else 0,
            'avg_fitness': np.mean([s.fitness for s in self.engine.population]) if self.engine.population else 0,
            'fitness_std': np.std([s.fitness for s in self.engine.population]) if self.engine.population else 0,
            
            # Diversity metrics
            'diversity': self.engine.state.diversity,
            'population_size': len(self.engine.population),
            'archive_size': len(self.engine.archive),
            
            # Innovation metrics
            'innovation_rate': self.engine.state.innovation_rate,
            'novelty_avg': np.mean([s.novelty for s in self.engine.population]) if self.engine.population else 0,
            'robustness_avg': np.mean([s.robustness for s in self.engine.population]) if self.engine.population else 0,
            
            # Quantum metrics
            'quantum_coherence': self.engine.state.quantum_coherence,
            'quantum_solutions': len([s for s in self.engine.population if s.quantum_state is not None]),
            'superposition_count': len([s for s in self.engine.population if s.superposition]),
            
            # Performance metrics
            'adaptation_speed': self.engine.state.adaptation_speed,
            'resilience_score': self.engine.state.resilience_score,
            'improvements_per_hour': self.engine.state.improvements_per_hour,
            
            # Evolution parameters
            'mutation_rate': self.engine.mutation_rate,
            'crossover_rate': self.engine.crossover_rate,
            'selection_pressure': self.engine.selection_pressure,
        }
        
        return metrics
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """Log evolution metrics"""
        # Simple console logging
        if self.evolution_cycles % 10 == 0:  # Log every 10 cycles
            print(f"\n📊 Evolution Metrics (Cycle {self.evolution_cycles})")
            print(f"   Generation: {metrics['generation']}")
            print(f"   Phase: {metrics['phase']}")
            print(f"   Best fitness: {metrics['best_fitness']:.4f}")
            print(f"   Diversity: {metrics['diversity']:.3f}")
            print(f"   Innovation rate: {metrics['innovation_rate']:.3f}")
            print(f"   Quantum coherence: {metrics['quantum_coherence']:.3f}")
        
        # Save to file
        try:
            with open(f"/tmp/l104_evolution_metrics_{datetime.now().date()}.jsonl", "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except:
            pass
    
    def _check_stagnation(self) -> bool:
        """Check if evolution is stagnating"""
        if len(self.engine.fitness_history) < 10:
            return False
        
        # Check last 10 generations for improvement
        recent_history = self.engine.fitness_history[-10:]
        improvements = [recent_history[i] - recent_history[i-1] for i in range(1, len(recent_history))]
        
        # Stagnation if average improvement < 0.001
        avg_improvement = np.mean(improvements) if improvements else 0
        return avg_improvement < 0.001
    
    def _check_convergence(self) -> bool:
        """Check for premature convergence"""
        return self.engine.state.diversity < 0.05 and self.engine.generation < 100
    
    def _should_adjust_parameters(self) -> bool:
        """Determine if evolution parameters should be adjusted"""
        # Adjust every 50 generations or when diversity is low
        return (self.engine.generation % 50 == 0 or 
                self.engine.state.diversity < 0.1)
    
    async def _trigger_innovation(self):
        """Trigger innovation phase"""
        print(f"   🚀 Triggering innovation phase")
        
        # Switch to innovation phase
        self.engine.state.phase = EvolutionPhase.INNOVATION
        self.engine.state.innovation_rate = 0.3
        
        # Increase mutation rate
        self.engine.mutation_rate = min(0.3, self.engine.mutation_rate * 1.5)
        
        # Introduce new random solutions
        if self.engine.population:
            # Replace worst 20% with random solutions
            replace_count = max(1, len(self.engine.population) // 5)
            
            # Keep track of parameter ranges from existing solutions
            param_ranges = {}
            for solution in self.engine.population:
                for key, value in solution.parameters.items():
                    if key not in param_ranges:
                        param_ranges[key] = [value, value]
                    else:
                        param_ranges[key][0] = min(param_ranges[key][0], value)
                        param_ranges[key][1] = max(param_ranges[key][1], value)
            
            # Create new random solutions
            for i in range(replace_count):
                idx = -i - 1  # Start from worst solutions
                if abs(idx) < len(self.engine.population):
                    # Create new solution
                    params = {}
                    for param_name, (min_val, max_val) in param_ranges.items():
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            params[param_name] = random.randint(int(min_val), int(max_val))
                        else:
                            params[param_name] = random.uniform(min_val, max_val)
                    
                    new_solution = EvolutionarySolution(
                        id=f"innov_{self.engine.generation}_{i:03d}",
                        parameters=params,
                        fitness=0.0,
                        novelty=1.0,  # Maximum novelty
                        robustness=0.8,
                    )
                    
                    # Add quantum properties
                    if random.random() < 0.5:
                        new_solution.quantum_state = np.random.randn(10)
                        new_solution.superposition = random.random() < 0.2
                    
                    self.engine.population[idx] = new_solution
        
        print(f"   ✅ Innovation triggered: {replace_count} new solutions introduced")
    
    async def _increase_diversity(self):
        """Increase population diversity"""
        print(f"   🌈 Increasing population diversity")
        
        # Increase mutation rate
        self.engine.mutation_rate = min(0.25, self.engine.mutation_rate * 1.3)
        
        # Decrease selection pressure
        self.engine.selection_pressure = max(0.05, self.engine.selection_pressure * 0.8)
        
        # Increase crossover rate
        self.engine.crossover_rate = min(0.9, self.engine.crossover_rate * 1.1)
        
        print(f"   ✅ Diversity measures applied")
        print(f"   • Mutation rate: {self.engine.mutation_rate:.3f}")
        print(f"   • Selection pressure: {self.engine.selection_pressure:.3f}")
        print(f"   • Crossover rate: {self.engine.crossover_rate:.3f}")
    
    async def _adjust_evolution_parameters(self):
        """Adjust evolution parameters based on performance"""
        print(f"   🔧 Adjusting evolution parameters")
        
        # Adjust based on phase
        if self.engine.state.phase == EvolutionPhase.EXPLORATION:
            # Higher mutation, lower selection pressure
            self.engine.mutation_rate = min(0.2, self.engine.mutation_rate * 1.1)
            self.engine.selection_pressure = max(0.1, self.engine.selection_pressure * 0.9)
            
        elif self.engine.state.phase == EvolutionPhase.EXPLOITATION:
            # Lower mutation, higher selection pressure
            self.engine.mutation_rate = max(0.05, self.engine.mutation_rate * 0.9)
            self.engine.selection_pressure = min(0.3, self.engine.selection_pressure * 1.1)
            
        elif self.engine.state.phase == EvolutionPhase.INNOVATION:
            # High mutation, medium selection
            self.engine.mutation_rate = min(0.3, self.engine.mutation_rate * 1.2)
            self.engine.selection_pressure = 0.15
            
        elif self.engine.state.phase == EvolutionPhase.CONVERGENCE:
            # Very low mutation, high selection
            self.engine.mutation_rate = max(0.01, self.engine.mutation_rate * 0.8)
            self.engine.selection_pressure = min(0.4, self.engine.selection_pressure * 1.2)
        
        # Adjust based on diversity
        if self.engine.state.diversity < 0.1:
            # Low diversity, increase exploration
            self.engine.mutation_rate = min(0.25, self.engine.mutation_rate * 1.15)
            self.engine.crossover_rate = min(0.85, self.engine.crossover_rate * 1.05)
            
        elif self.engine.state.diversity > 0.3:
            # High diversity, increase exploitation
            self.engine.mutation_rate = max(0.05, self.engine.mutation_rate * 0.9)
            self.engine.selection_pressure = min(0.35, self.engine.selection_pressure * 1.1)
        
        print(f"   ✅ Parameters adjusted")
        print(f"   • Phase: {self.engine.state.phase.value}")
        print(f"   • Mutation: {self.engine.mutation_rate:.3f}")
        print(f"   • Selection: {self.engine.selection_pressure:.3f}")
        print(f"   • Crossover: {self.engine.crossover_rate:.3f}")
        print(f"   • Diversity: {self.engine.state.diversity:.3f}")
    
    def _analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends over time"""
        trends = {}
        
        if len(self.engine.fitness_history) > 10:
            # Fitness trend (last 10 vs previous 10)
            recent = self.engine.fitness_history[-10:]
            previous = self.engine.fitness_history[-20:-10] if len(self.engine.fitness_history) >= 20 else recent
            
            trends['fitness_trend'] = np.mean(recent) - np.mean(previous)
            
            # Diversity trend
            # Note: diversity history would need to be tracked separately
            trends['diversity_trend'] = 0  # Placeholder
            
            # Innovation trend
            if len(self.engine.innovation_history) > 10:
                recent_innov = self.engine.innovation_history[-10:]
                previous_innov = self.engine.innovation_history[-20:-10] if len(self.engine.innovation_history) >= 20 else recent_innov
                trends['innovation_trend'] = np.mean(recent_innov) - np.mean(previous_innov)
        
        return trends
    
    async def _optimize_for_speed(self):
        """Optimize evolution for faster improvement"""
        print(f"   ⚡ Optimizing for speed")
        
        # Increase selection pressure
        self.engine.selection_pressure = min(0.3, self.engine.selection_pressure * 1.2)
        
        # Slightly increase mutation
        self.engine.mutation_rate = min(0.15, self.engine.mutation_rate * 1.1)
        
        # Reduce elitism (