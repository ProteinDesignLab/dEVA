class Individual(object):
    def __init__(self, generation=0, index=0):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.fitnesses = {}
        self.generation = generation
        self.index = index
        self.name = None
        
        self.sequence = None
        self.sequence_ = None

    def update_fitness(self, prob):
        for key, value in prob.items():
            if key not in self.fitnesses.keys():
                raise ValueError(f'Key for {key} is not in fitness_keys.')
            self.fitnesses[key] = value

    def add_fitness(self, prob):
        for key, value in prob.items():
            if key in self.fitnesses.keys():
                raise ValueError(f'Fitness for {key} already added.')
            self.fitnesses[key] = value
    
    def get_gen(self):
        return self.generation
    
    def get_index(self):
        return self.index
    
    def get_name(self):
        return self.name
    
    def update_seq_str(self, seq_str):
        self.sequence = seq_str

    def update_seq_tensor(self, seq_tensor):
        self.sequence_ = seq_tensor

    def update_name(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.fitnesses == other.fitnesses
        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for objective in self.fitnesses.keys():
            and_condition = and_condition and self.fitnesses[objective] >= other_individual.fitnesses[objective]
            or_condition = or_condition or self.fitnesses[objective] > other_individual.fitnesses[objective]
        return (and_condition and or_condition)

    
    def add_header(self, filename):

        header_lines = []
        for key, value in self.fitnesses.items():
            header_lines.append(f"REMARK 220 REMARK: {key} = {value}")
        
        header_lines.append(f"REMARK 220 REMARK: sequence = {self.sequence}")
        header_lines.append("REMARK 220 dEVA-generated design")
        
        header = "\n".join(header_lines) + "\n"
        
        with open(filename) as f:
            lines = f.readlines()
        with open(filename, "w") as f:
            f.write(header)
            f.writelines(lines)