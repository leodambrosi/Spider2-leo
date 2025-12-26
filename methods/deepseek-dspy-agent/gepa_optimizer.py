"""
GEPA (Genetic Evolutionary Program Augmentation) optimizer for DSPy.

This module implements GEPA optimization for DSPy programs.
GEPA uses genetic algorithms to evolve and optimize prompt programs.
"""

import random
from typing import List, Tuple, Callable, Optional, Dict, Any
import dspy
from dspy.teleprompt import Teleprompter


class GEPAOptimizer(Teleprompter):
    """
    Genetic Evolutionary Program Augmentation optimizer for DSPy.

    GEPA evolves a population of prompt programs using genetic operations:
    - Crossover: Combine two parent programs
    - Mutation: Randomly modify program components
    - Selection: Select best programs based on fitness
    """

    def __init__(
        self,
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
        metric: Optional[Callable] = None,
        **kwargs
    ):
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.metric = metric or self._default_metric
        self.kwargs = kwargs

    def _default_metric(self, example, pred, trace=None) -> float:
        """Default metric: simple correctness check."""
        # This should be replaced with task-specific metric
        return 1.0 if pred and hasattr(pred, 'score') else 0.0

    def _initialize_population(self, program: dspy.Module) -> List[dspy.Module]:
        """Initialize population with variations of the base program."""
        population = []

        # Add the original program
        population.append(program)

        # Create variations
        for i in range(self.population_size - 1):
            variant = self._mutate_program(program.deepcopy(), rate=0.3)
            population.append(variant)

        return population

    def _mutate_text(self, text: str) -> str:
        """Apply random mutations to text."""
        if not text:
            return text

        words = text.split()
        if len(words) <= 1:
            return text

        # Random mutation operations
        mutation_type = random.choice(['replace', 'delete', 'insert', 'swap'])

        if mutation_type == 'replace' and len(words) > 1:
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(['important', 'critical', 'key', 'essential', 'relevant',
                                         'significant', 'crucial', 'vital', 'major', 'primary',
                                         'fundamental', 'central', 'necessary', 'indispensable'])

        elif mutation_type == 'delete' and len(words) > 2:
            idx = random.randint(0, len(words) - 1)
            del words[idx]

        elif mutation_type == 'insert' and len(words) >= 1:
            idx = random.randint(0, len(words))
            words.insert(idx, random.choice(['carefully', 'precisely', 'accurately', 'thoroughly',
                                              'systematically', 'methodically', 'rigorously',
                                              'efficiently', 'effectively', 'strategically']))

        elif mutation_type == 'swap' and len(words) > 1:
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]

        return ' '.join(words)

    def _mutate_program(self, program: dspy.Module, rate: float = None) -> dspy.Module:
        """Apply mutations to a program."""
        mutation_rate = rate or self.mutation_rate

        # Mutate signature instructions if available
        if hasattr(program, 'signature') and random.random() < mutation_rate:
            signature = program.signature
            # Try to mutate signature docstring/instructions
            if hasattr(signature, '__doc__') and signature.__doc__:
                signature.__doc__ = self._mutate_text(signature.__doc__)
            # Try to mutate signature fields' descriptions
            if hasattr(signature, 'fields'):
                for field_name, field in signature.fields.items():
                    if hasattr(field, 'desc') and field.desc:
                        field.desc = self._mutate_text(field.desc)

        # Mutate demonstrations if available
        if hasattr(program, 'demonstrations') and random.random() < mutation_rate:
            demos = program.demonstrations
            if demos and len(demos) > 0:
                # Randomly modify one demonstration
                idx = random.randint(0, len(demos) - 1)
                demo = demos[idx]
                # Simple mutation: could modify demo attributes
                # For now, just mark as modified (placeholder)
                # In practice, would mutate demo inputs/outputs
                pass

        # Mutate demos attribute (common in DSPy)
        if hasattr(program, 'demos') and random.random() < mutation_rate:
            demos = program.demos
            if demos and len(demos) > 0:
                # Similar to above
                pass

        return program

    def _crossover_programs(self, parent1: dspy.Module, parent2: dspy.Module) -> dspy.Module:
        """Create child program by crossover of two parents."""
        child = parent1.deepcopy()

        # Crossover signature instructions
        if (hasattr(child, 'signature') and hasattr(child.signature, '__doc__') and
            hasattr(parent2, 'signature') and hasattr(parent2.signature, '__doc__')):
            if random.random() < 0.5:
                child.signature.__doc__ = parent2.signature.__doc__

        # Crossover signature field descriptions
        if (hasattr(child, 'signature') and hasattr(child.signature, 'fields') and
            hasattr(parent2, 'signature') and hasattr(parent2.signature, 'fields')):
            for field_name, child_field in child.signature.fields.items():
                if (field_name in parent2.signature.fields and
                    hasattr(child_field, 'desc') and hasattr(parent2.signature.fields[field_name], 'desc')):
                    if random.random() < 0.3:  # 30% chance to copy field description
                        child_field.desc = parent2.signature.fields[field_name].desc

        # Crossover demonstrations
        if hasattr(parent1, 'demonstrations') and hasattr(parent2, 'demonstrations'):
            if parent2.demonstrations and random.random() < 0.5:
                child.demonstrations = parent2.demonstrations.copy()

        # Crossover demos (common attribute name)
        if hasattr(parent1, 'demos') and hasattr(parent2, 'demos'):
            if parent2.demos and random.random() < 0.5:
                child.demos = parent2.demos.copy()

        return child

    def _evaluate_population(
        self,
        population: List[dspy.Module],
        trainset: List[dspy.Example]
    ) -> List[Tuple[dspy.Module, float]]:
        """Evaluate fitness of each program in population."""
        evaluated = []

        for program in population:
            total_score = 0.0
            # Evaluate on subset for efficiency
            eval_size = min(5, len(trainset))
            eval_examples = random.sample(trainset, eval_size) if len(trainset) > eval_size else trainset

            for example in eval_examples:
                try:
                    pred = program(**example.inputs())
                    score = self.metric(example, pred)
                    total_score += score
                except Exception as e:
                    total_score += 0.0

            avg_score = total_score / len(eval_examples) if eval_examples else 0.0
            evaluated.append((program, avg_score))

        # Sort by fitness (descending)
        evaluated.sort(key=lambda x: x[1], reverse=True)
        return evaluated

    def compile(
        self,
        program: dspy.Module,
        trainset: List[dspy.Example],
        valset: Optional[List[dspy.Example]] = None
    ) -> dspy.Module:
        """
        Compile program using GEPA optimization.

        Args:
            program: Initial DSPy program
            trainset: Training examples
            valset: Validation examples (optional)

        Returns:
            Optimized DSPy program
        """
        # Initialize population
        population = self._initialize_population(program)

        # Evolutionary loop
        for generation in range(self.generations):
            # Evaluate population
            evaluated = self._evaluate_population(population, trainset)

            # Select elites
            elites = [prog for prog, score in evaluated[:self.elite_size]]

            # Create next generation
            next_generation = elites.copy()

            # Generate offspring
            while len(next_generation) < self.population_size:
                # Selection: tournament selection
                tournament_size = 3
                tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
                parent1 = max(tournament, key=lambda x: x[1])[0]

                tournament = random.sample(evaluated, min(tournament_size, len(evaluated)))
                parent2 = max(tournament, key=lambda x: x[1])[0]

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover_programs(parent1, parent2)
                else:
                    child = parent1.deepcopy()

                # Mutation
                child = self._mutate_program(child)

                next_generation.append(child)

            population = next_generation

            # Log best fitness
            best_score = evaluated[0][1]
            print(f"Generation {generation + 1}/{self.generations}, Best fitness: {best_score:.3f}")

        # Return best program
        final_evaluated = self._evaluate_population(population, trainset)
        best_program = final_evaluated[0][0]

        return best_program


# Convenience function for using GEPA
def GEPA(
    metric: Optional[Callable] = None,
    population_size: int = 20,
    generations: int = 10,
    **kwargs
) -> GEPAOptimizer:
    """Create a GEPA optimizer with given parameters."""
    return GEPAOptimizer(
        population_size=population_size,
        generations=generations,
        metric=metric,
        **kwargs
    )