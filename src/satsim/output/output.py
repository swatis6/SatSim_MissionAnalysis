from satsim.output.visualizer import Visualizer


class Output:
    #plots, data exports, reports

    def __init__(self, history, config): #history is the sim.history dict, config is output_config from yaml
        
        self.history = history
        self.config = config.get("output", {})


        self.visualizer = Visualizer(
            history,
            self.config.get("visualization", {})
        )

    def run(self):
        self.visualizer.run()

        # Future:
        # if self.config.get("save_data", False):
        #     self.save_data()
        # if self.config.get("generate_report", False):
        #     self.generate_report()