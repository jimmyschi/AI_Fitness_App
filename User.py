class User:
    #TODO: get body_mass, exercise, and video from UI
    def __init__(self, body_mass, exercise, video) -> None:
        self.body_mass = 200
        self.exercise = exercise
        self.video = video
        self.exercise_selection = ["curl", "bench", "incline_bench", 
                                "shoulder_press","lat_pulldown", "lat_raise",
                                "pull-up", "dip", "t_bar_row", "squat", "pec_deck", "rear_delt",
                                "deadlift", "pushdown", "push_up", "hamstring_curl",
                                "leg_extension", "front_raise", "overhead_tricep", "kickback"]
    