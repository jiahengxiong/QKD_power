global protocol
global detector
global ice_box_capacity
global bypass
global key_rate_list


cases = [
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Paris", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Tokyo", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Large", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": True,  "Detector": "APD",       "Topology": "Large", "Traffic": "High"},

    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Large", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": True,  "Detector": "SNSPD",     "Topology": "Large", "Traffic": "High"},

    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Paris", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Tokyo", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Large", "Traffic": "Medium"},
    # {"Protocol": "BB84", "Bypass": False, "Detector": "APD",       "Topology": "Large", "Traffic": "High"},

    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Large", "Traffic": "Medium"},
    {"Protocol": "BB84", "Bypass": False, "Detector": "SNSPD",     "Topology": "Large", "Traffic": "High"},

    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": True,  "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "High"},

    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Paris", "Traffic": "High"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Tokyo", "Traffic": "High"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "Low"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "Medium"},
    {"Protocol": "CV-QKD", "Bypass": False, "Detector": "ThorlabsPDB", "Topology": "Large", "Traffic": "High"}
]
Traffic_cases = {'Paris':{'Low': 50000, 'Medium':100000, 'High':400000},
                 'Tokyo':{'Low': 600000, 'Medium':1200000, 'High':4320000},
                 'Large': {'Low': 15000, 'Medium': 30000, 'High': 100000}}
