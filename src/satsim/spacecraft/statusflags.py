class StatusFlags:
    def __init__(self):
        #mission phases
        self.deployed = False
        self.launch_phase = True
        self.detumbling = False
        self.nominal_ops = False
        self.safe_mode = False
        self.end_of_mission = False
        self.decaying = False  

        #adcs stuff
        self.adcs_initialized = False
        self.attitude_valid = False
        self.sun_pointing = False
        self.nadir_pointing = False

        #comms stuff
        self.comms_initialized = False
        self.in_contact = False
        self.downlink_active = False
        self.uplink_active = False

        #sim control
        
        self.initialized = True
        self.crashed = False