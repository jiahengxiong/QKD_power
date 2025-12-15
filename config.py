global protocol
global detector
global ice_box_capacity
global bypass
global key_rate_list
# cases = [{'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#          {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'}]


cases = [
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'}
   ]
# cases = [{'Topology': 'Test', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'}]
# cases = [{'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#          {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'}]

# cases = [
#     # Tokyo 拓扑组
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'}
# ]

"""cases = [
    # Paris 拓扑组
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'}
]"""
"""cases = [
    # Paris 拓扑组
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Paris', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},

    # Tokyo 拓扑组
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},

    # Large 拓扑组
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
    {'Topology': 'Large', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'}
]"""
"""cases = [
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
]"""
Traffic_cases = {'Paris':{'Low': 50000, 'Medium':100000, 'High':400000},
                 'Tokyo':{'Low': 350000, 'Medium':700000, 'High':2000000},
                 'Large': {'Low': 12500, 'Medium': 25000, 'High': 90000}}


# cases = [
#     # Tokyo 拓扑组
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': True, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'CV-QKD', 'Bypass': False, 'Detector': 'ThorlabsPDB', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'APD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': True, 'Detector': 'SNSPD', 'Traffic': 'High'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'APD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Low'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'Medium'},
#     {'Topology': 'Tokyo', 'Protocol': 'BB84', 'Bypass': False, 'Detector': 'SNSPD', 'Traffic': 'High'}
# ]