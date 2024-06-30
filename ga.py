from typing import List, Tuple, Optional

class GeneticApplier:
    def __init__(self):
        self.prev_best_solution = None
        self.last_clicked_cell = None
        self.genetic_listener = None

    def predict(self, N: int, board: List[List[MyColor]], last_clicked_cell: 'CellState'):
        self.N = N
        self.board = board
        self.last_clicked_cell = last_clicked_cell

        if self.is_prev_solution_winnable(self.prev_best_solution, board):
            self.genetic_listener.on_progress(100)
            self.genetic_listener.on_draw_request(self.prev_best_solution)

            to_place = self.choose_the_best_cell_to_place(self.prev_best_solution)
            if to_place is None:
                if self.genetic_listener:
                    self.genetic_listener.on_error("Something went wrong", True)
                return

            if self.genetic_listener:
                self.genetic_listener.on_finished(to_place)
            return

        populations = self.init_population(board)
        if populations is None:
            return

        global_best = self.get_the_best(populations)
        global_best_val = self.calc_fitness(board, global_best)

        it = 0
        while it < self.NO_OF_IT:
            it += 1
            offspring = []

            while len(offspring) < len(populations):
                parent_one = self.get_parent_from_tournament(populations)
                parent_two = self.get_parent_from_tournament(populations)
                child_after_cross = self.apply_crossover(parent_one, parent_two)

                for child in child_after_cross:
                    mutated_child = self.mutate_children(child)
                    offspring.append(mutated_child)

            local_best = self.get_the_best(offspring)
            local_best_val = self.calc_fitness(board, local_best)

            if local_best_val > global_best_val:
                global_best_val = local_best_val
                global_best = local_best

            progress = (100 * it) // self.NO_OF_IT
            self.genetic_listener.on_progress(progress)

        self.prev_best_solution = global_best
        self.genetic_listener.on_draw_request(global_best)

        to_place = self.choose_the_best_cell_to_place(global_best)
        if to_place is None:
            if self.genetic_listener:
                self.genetic_listener.on_error("Something went wrong", True)
            return

        if self.genetic_listener:
            self.genetic_listener.on_finished(to_place)
