import uuid

# import matplotlib.pyplot as plt
import networkx as nx

from QEnergy.studies.key_rate_compute import compute_key_rate
import plotly.graph_objects as go


class Network:
    def __init__(self, map_name, wavelength_list, protocol, receiver):
        self.num_wavelength = 0
        self.name = f"{map_name} Network"
        self.wavelength_list = wavelength_list
        self.protocol = protocol
        self.receiver = receiver

        self.topology = self.get_topology(map_name)
        self.physical_topology = self.get_physical_topology(map_name)

    def get_physical_topology(self, map_name):
        G = nx.Graph()
        if map_name == "Test":
            edges = [(0, 1, 10),
                     (1, 2, 10),]
            for node1, node2, distance in edges:
                G.add_edge(node1, node2, distance=distance, type='fiber')
        if map_name == "Tokyo":
            edges = [
                ("Setagaya", "Ota", 12.045),
                ("Setagaya", "Shinagawa", 9.072),
                ("Setagaya", "Minato", 9.945),
                ("Setagaya", "Shinjuku", 7.916),
                ("Setagaya", "Nerima", 10.885),
                ("Ota", "Shinagawa", 6.464),
                ("Shinagawa", "Minato", 6.788),
                ("Minato", "Koto", 7.195),
                ("Minato", "Chiyoda", 4.972),
                ("Shinjuku", "Minato", 6.878),
                ("Shinjuku", "Chiyoda", 5.461),
                ("Shinjuku", "Itabashi", 7.446),
                ("Shinjuku", "Nerima", 7.504),
                ("Nerima", "Itabashi", 6.341),
                ("Koto", "Edogawa", 6.868),
                ("Chiyoda", "Koto", 7.248),
                ("Chiyoda", "Edogawa", 11.528),
                ("Chiyoda", "Bunkyo", 2.601),
                ("Bunkyo", "Adachi", 9.701),
                ("Itabashi", "Bunkyo", 7.239),
                ("Edogawa", "Adachi", 10.682)
            ]
            for node1, node2, distance in edges:
                G.add_edge(node1, node2, distance=distance, type='fiber')
        if map_name == "Paris":
            edges = [
                ("LKB-2", "LKB", 0.027),
                ("LKB", "WL", 0.176),
                ("WL", "WL-2", 0.025),
                ("LKB", "LIP6", 0.206),
                ("LIP6", "LIP6-2", 0.093),
                ("LKB", "LIP6-2", 0.190),
                ("LIP6", "MPQ", 13.245),
                ("LIP6", "OG", 13.991),
                ("OG", "TP", 43.0),
                ("TP", "TRT", 2.0),
                ("TRT", "IOGS", 0.97),
                ("WL", "LIP6-2", 0.264)
            ]
            for node1, node2, distance in edges:
                G.add_edge(node1, node2, distance=distance, type='fiber')
        if map_name == "G50":
            edge_list = [(1, 2), (1, 5), (2, 7), (6, 7), (3, 7), (2, 6), (4, 8), (5, 9), (9, 11), (6, 11), (6, 12),
                         (7, 13), (7, 14), (3, 14), (4, 17), (8, 17), (8, 10), (10, 15), (10, 11), (11, 12), (12, 13),
                         (13, 14), (15, 16), (12, 16), (15, 19), (16, 28), (12, 20), (13, 21), (14, 21), (14, 25),
                         (11, 16), (17, 18), (18, 22), (22, 27), (17, 26), (26, 27), (19, 20), (20, 23), (21, 23),
                         (23, 25), (23, 24), (24, 25), (21, 25), (21, 33), (24, 33), (19, 28), (28, 29), (20, 29),
                         (20, 30), (29, 30), (28, 31), (27, 31), (29, 32), (30, 32), (30, 36), (31, 32), (32, 35),
                         (26, 34), (31, 34), (31, 39), (35, 39), (38, 39), (34, 38), (38, 41), (39, 41), (40, 41),
                         (35, 40), (23, 36), (36, 37), (33, 37), (36, 42), (36, 45), (37, 46), (37, 43), (43, 47),
                         (43, 46), (46, 47), (45, 46), (46, 50), (49, 50), (48, 49), (41, 48), (41, 42), (42, 44),
                         (42, 49), (44, 45), (18, 19), (8, 9)]
            length_list = [68.0, 148.0, 114.0, 94.0, 140.0, 86.0, 84.0, 55.0, 100.0, 132.0, 148.0, 167.0, 181.0, 175.0,
                           219.0, 197.0, 97.0, 47.0, 114.0, 55.0, 77.0, 128.0, 63.0, 138.0, 50.0, 131.0, 126.0, 101.0,
                           150.0, 165.0, 91.0, 35.0, 32.0, 33.0, 105.0, 64.0, 143.0, 113.0, 102.0, 190.0, 134.0, 62.0,
                           100.0, 165.0, 138.0, 81.0, 56.0, 100.0, 86.0, 71.0, 65.0, 79.0, 52.0, 82.0, 86.0, 82.0, 27.0,
                           120.0, 95.0, 102.0, 80.0, 60.0, 64.0, 104.0, 66.0, 53.0, 44.0, 152.0, 91.0, 66.0, 125.0,
                           173.0, 151.0, 89.0, 109.0, 105.0, 145.0, 56.0, 104.0, 86.0, 105.0, 120.0, 62.0, 73.0, 124.0,
                           67.0, 32.0, 68.0]

            for i in range(len(edge_list)):
                G.add_edge(edge_list[i][0], edge_list[i][1], distance=length_list[i], type='fiber')

        if map_name == "Large":
            """edge_list = [(1, 2), (1, 5), (2, 7), (6, 7), (3, 7), (2, 6), (4, 8), (5, 9), (9, 11), (6, 11), (6, 12),
                         (7, 13),
                         (7, 14), (3, 14), (4, 17), (8, 17), (8, 10), (10, 15), (10, 11), (11, 12), (12, 13), (13, 14),
                         (15, 16), (12, 16), (15, 19), (16, 28), (12, 20), (13, 21), (14, 21), (14, 25), (11, 16),
                         (17, 18),
                         (18, 22), (22, 27), (17, 26), (26, 27), (19, 20), (20, 23), (21, 23), (23, 25), (23, 24),
                         (24, 25),
                         (21, 25), (21, 33), (24, 33), (19, 28), (28, 29), (20, 29), (20, 30), (29, 30), (28, 31),
                         (27, 31),
                         (29, 32), (30, 32), (30, 36), (31, 32), (32, 35), (26, 34), (31, 34), (31, 39), (35, 39),
                         (38, 39),
                         (34, 38), (38, 41), (39, 41), (40, 41), (35, 40), (23, 36), (36, 37), (33, 37), (36, 42),
                         (36, 45),
                         (37, 46), (37, 43), (43, 47), (43, 46), (46, 47), (45, 46), (46, 50), (49, 50), (48, 49),
                         (41, 48),
                         (41, 42), (42, 44), (42, 49), (44, 45), (18, 19), (8, 9)]
            length_list = [68, 148, 114, 94, 140, 86, 84, 55, 100, 132, 148, 167, 181, 175, 219, 197, 97, 47, 114, 55,
                           77, 128,
                           63, 138, 50, 131, 126, 101, 150, 165, 91, 35, 32, 33, 105, 64, 143, 113, 102, 190, 134, 62,
                           100, 165,
                           138, 81, 56, 100, 86, 71, 65, 79, 52, 82, 86, 82, 27, 120, 95, 102, 80, 60, 64, 104, 66, 53,
                           44, 152,
                           91, 66, 125, 173, 151, 89, 109, 105, 145, 56, 104, 86, 105, 120, 62, 73, 124, 67, 32, 68]"""
            edge_list = [
                (1, 2), (1, 6),
                (2, 3), (2, 6),
                (3, 4), (3, 5), (3, 7),
                (4, 5), (4, 7),
                (5, 8),
                (6, 7), (6, 9), (6, 11),
                (7, 8), (7, 9),
                (8, 10),
                (9, 10), (9, 12), (9, 11),
                (10, 13), (10, 14),
                (11, 15), (11, 19), (11, 12),
                (12, 13), (12, 16),
                (13, 14), (13, 17),
                (14, 18),
                (15, 16), (15, 20),
                (16, 17), (16, 21), (16, 22),
                (17, 18), (17, 22), (17, 23),
                (18, 24),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24)
            ]
            length_list = [800, 1000, 1100, 950, 250, 1000, 1000, 800, 850, 1200, 1000, 1200, 1900, 1150, 1000, 900,
                           1000, 1000,
                           1400, 950, 850, 1300, 2800, 900, 800, 1000, 650, 800, 1200, 800, 1300, 800, 1000, 800, 800,
                           850, 1000,
                           900, 1200, 700, 300, 600, 900]

            # 缩放到 [30, 130]
            min_val = 30
            max_val = 130

            # 数据缩放公式
            scaled_list = [
                min_val + (x - min(length_list)) * (max_val - min_val) / (max(length_list) - min(length_list)) for x in
                length_list]

            # 转换为整数
            length_list = [int(round(x)) for x in scaled_list]
            for i in range(len(edge_list)):
                G.add_edge(edge_list[i][0], edge_list[i][1], distance=length_list[i], type='fiber')
        return G

    def get_topology(self, map_name):
        G = nx.MultiGraph()
        if map_name == "Test":
            edges = [(0, 1, 10),
                     (1, 2, 10),]
            for node1, node2, distance in edges:
                for wavelength in self.wavelength_list:
                    self.num_wavelength += 1
                    G.add_edge(node1, node2, distance=distance, key=uuid.uuid4().hex,
                               wavelength=wavelength, laser=0, detector=0,
                               capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                         receiver=self.receiver),
                               used_capacity=0, occupied=False,
                               free_capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                              receiver=self.receiver))
            for node in G.nodes:
                G.nodes[node]['laser'] = {}
                G.nodes[node]['laser_capacity'] = {}
                G.nodes[node]['ice_box'] = 0
                G.nodes[node]['num_detector'] = 0
                for wavelength in self.wavelength_list:
                    G.nodes[node]['laser'][wavelength] = []
                    G.nodes[node]['laser_capacity'][wavelength] = {}
                    G.nodes[node]['detector'] = []
        if map_name == "Tokyo":
            self.num_wavelength = 0
            edges = [
                ("Setagaya", "Ota", 12.045),
                ("Setagaya", "Shinagawa", 9.072),
                ("Setagaya", "Minato", 9.945),
                ("Setagaya", "Shinjuku", 7.916),
                ("Setagaya", "Nerima", 10.885),
                ("Ota", "Shinagawa", 6.464),
                ("Shinagawa", "Minato", 6.788),
                ("Minato", "Koto", 7.195),
                ("Minato", "Chiyoda", 4.972),
                ("Shinjuku", "Minato", 6.878),
                ("Shinjuku", "Chiyoda", 5.461),
                ("Shinjuku", "Itabashi", 7.446),
                ("Shinjuku", "Nerima", 7.504),
                ("Nerima", "Itabashi", 6.341),
                ("Koto", "Edogawa", 6.868),
                ("Chiyoda", "Koto", 7.248),
                ("Chiyoda", "Edogawa", 11.528),
                ("Chiyoda", "Bunkyo", 2.601),
                ("Bunkyo", "Adachi", 9.701),
                ("Itabashi", "Bunkyo", 7.239),
                ("Edogawa", "Adachi", 10.682)
            ]
            for node1, node2, distance in edges:
                for wavelength in self.wavelength_list:
                    self.num_wavelength += 1
                    G.add_edge(node1, node2, distance=distance, key=uuid.uuid4().hex,
                               wavelength=wavelength, laser=0, detector=0,
                               capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                         receiver=self.receiver),
                               used_capacity=0, occupied=False,
                               free_capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                              receiver=self.receiver))

            for node in G.nodes:
                G.nodes[node]['laser'] = {}
                G.nodes[node]['detector'] = {}
                G.nodes[node]['laser_capacity'] = {}
                G.nodes[node]['ice_box'] = 0
                G.nodes[node]['num_detector'] = 0
                for wavelength in self.wavelength_list:
                    G.nodes[node]['laser'][wavelength] = []
                    G.nodes[node]['laser_capacity'][wavelength] = {}
                    G.nodes[node]['detector'][wavelength] = []
        if map_name == "Paris":
            self.num_wavelength = 0
            edges = [
                ("LKB-2", "LKB", 0.027),
                ("LKB", "WL", 0.176),
                ("WL", "WL-2", 0.025),
                ("LKB", "LIP6", 0.206),
                ("LIP6", "LIP6-2", 0.093),
                ("LKB", "LIP6-2", 0.190),
                ("LIP6", "MPQ", 13.245),
                ("LIP6", "OG", 13.991),
                ("OG", "TP", 43.0),
                ("TP", "TRT", 2.0),
                ("TRT", "IOGS", 0.97),
                ("WL", "LIP6-2", 0.264)
            ]
            for node1, node2, distance in edges:
                for wavelength in self.wavelength_list:
                    self.num_wavelength += 1
                    G.add_edge(node1, node2, distance=distance, key=uuid.uuid4().hex,
                               wavelength=wavelength, laser=0, detector=0,
                               capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                         receiver=self.receiver),
                               used_capacity=0, occupied=False,
                               free_capacity=compute_key_rate(distance=distance, protocol=self.protocol,
                                                              receiver=self.receiver))
            for node in G.nodes:
                G.nodes[node]['laser'] = {}
                G.nodes[node]['detector'] = {}
                G.nodes[node]['laser_capacity'] = {}
                G.nodes[node]['ice_box'] = 0
                G.nodes[node]['num_detector'] = 0
                for wavelength in self.wavelength_list:
                    G.nodes[node]['laser'][wavelength] = []
                    G.nodes[node]['laser_capacity'][wavelength] = {}
                    G.nodes[node]['detector'][wavelength] = []
        if map_name == "G50":
            edge_list = [(1, 2), (1, 5), (2, 7), (6, 7), (3, 7), (2, 6), (4, 8), (5, 9), (9, 11), (6, 11), (6, 12),
                         (7, 13), (7, 14), (3, 14), (4, 17), (8, 17), (8, 10), (10, 15), (10, 11), (11, 12), (12, 13),
                         (13, 14), (15, 16), (12, 16), (15, 19), (16, 28), (12, 20), (13, 21), (14, 21), (14, 25),
                         (11, 16), (17, 18), (18, 22), (22, 27), (17, 26), (26, 27), (19, 20), (20, 23), (21, 23),
                         (23, 25), (23, 24), (24, 25), (21, 25), (21, 33), (24, 33), (19, 28), (28, 29), (20, 29),
                         (20, 30), (29, 30), (28, 31), (27, 31), (29, 32), (30, 32), (30, 36), (31, 32), (32, 35),
                         (26, 34), (31, 34), (31, 39), (35, 39), (38, 39), (34, 38), (38, 41), (39, 41), (40, 41),
                         (35, 40), (23, 36), (36, 37), (33, 37), (36, 42), (36, 45), (37, 46), (37, 43), (43, 47),
                         (43, 46), (46, 47), (45, 46), (46, 50), (49, 50), (48, 49), (41, 48), (41, 42), (42, 44),
                         (42, 49), (44, 45), (18, 19), (8, 9)]
            length_list = [68.0, 148.0, 114.0, 94.0, 140.0, 86.0, 84.0, 55.0, 100.0, 132.0, 148.0, 167.0, 181.0, 175.0,
                           219.0, 197.0, 97.0, 47.0, 114.0, 55.0, 77.0, 128.0, 63.0, 138.0, 50.0, 131.0, 126.0, 101.0,
                           150.0, 165.0, 91.0, 35.0, 32.0, 33.0, 105.0, 64.0, 143.0, 113.0, 102.0, 190.0, 134.0, 62.0,
                           100.0, 165.0, 138.0, 81.0, 56.0, 100.0, 86.0, 71.0, 65.0, 79.0, 52.0, 82.0, 86.0, 82.0, 27.0,
                           120.0, 95.0, 102.0, 80.0, 60.0, 64.0, 104.0, 66.0, 53.0, 44.0, 152.0, 91.0, 66.0, 125.0,
                           173.0, 151.0, 89.0, 109.0, 105.0, 145.0, 56.0, 104.0, 86.0, 105.0, 120.0, 62.0, 73.0, 124.0,
                           67.0, 32.0, 68.0]

            for i in range(len(edge_list)):
                for wavelength in self.wavelength_list:
                    self.num_wavelength += 1
                    G.add_edge(edge_list[i][0], edge_list[i][1], distance=length_list[i], key=uuid.uuid4().hex,
                               wavelength=wavelength, laser=0, detector=0,
                               capacity=compute_key_rate(distance=length_list[i], protocol=self.protocol,
                                                         receiver=self.receiver),
                               used_capacity=0, occupied=False,
                               free_capacity=compute_key_rate(distance=length_list[i], protocol=self.protocol,
                                                              receiver=self.receiver)
                               )
            for node in G.nodes:
                G.nodes[node]['laser'] = {}
                G.nodes[node]['detector'] = {}
                G.nodes[node]['laser_capacity'] = {}
                G.nodes[node]['ice_box'] = 0
                G.nodes[node]['num_detector'] = 0
                for wavelength in self.wavelength_list:
                    G.nodes[node]['laser'][wavelength] = []
                    G.nodes[node]['laser_capacity'][wavelength] = {}
                    G.nodes[node]['detector'][wavelength] = []
        if map_name == "Large":
            self.num_wavelength = 0
            edge_list = [
                (1, 2), (1, 6),
                (2, 3), (2, 6),
                (3, 4), (3, 5), (3, 7),
                (4, 5), (4, 7),
                (5, 8),
                (6, 7), (6, 9), (6, 11),
                (7, 8), (7, 9),
                (8, 10),
                (9, 10), (9, 12), (9, 11),
                (10, 13), (10, 14),
                (11, 15), (11, 19), (11, 12),
                (12, 13), (12, 16),
                (13, 14), (13, 17),
                (14, 18),
                (15, 16), (15, 20),
                (16, 17), (16, 21), (16, 22),
                (17, 18), (17, 22), (17, 23),
                (18, 24),
                (19, 20),
                (20, 21),
                (21, 22),
                (22, 23),
                (23, 24)
            ]
            length_list = [800, 1000, 1100, 950, 250, 1000, 1000, 800, 850, 1200, 1000, 1200, 1900, 1150, 1000, 900,
                           1000, 1000,
                           1400, 950, 850, 1300, 2800, 900, 800, 1000, 650, 800, 1200, 800, 1300, 800, 1000, 800, 800,
                           850, 1000,
                           900, 1200, 700, 300, 600, 900]

            # 缩放到 [30, 130]
            min_val = 30
            max_val = 130

            # 数据缩放公式
            scaled_list = [
                min_val + (x - min(length_list)) * (max_val - min_val) / (max(length_list) - min(length_list)) for x in
                length_list]

            # 转换为整数
            length_list = [int(round(x)) for x in scaled_list]
            for i in range(len(edge_list)):
                for wavelength in self.wavelength_list:
                    self.num_wavelength += 1
                    G.add_edge(edge_list[i][0], edge_list[i][1], distance=length_list[i], key=uuid.uuid4().hex,
                               wavelength=wavelength, laser=0, detector=0,
                               capacity=compute_key_rate(distance=length_list[i], protocol=self.protocol,
                                                         receiver=self.receiver),
                               used_capacity=0, occupied=False,
                               free_capacity=compute_key_rate(distance=length_list[i], protocol=self.protocol,
                                                              receiver=self.receiver)
                               )
            for node in G.nodes:
                G.nodes[node]['laser'] = {}
                G.nodes[node]['detector'] = {}
                G.nodes[node]['laser_capacity'] = {}
                G.nodes[node]['ice_box'] = 0
                G.nodes[node]['num_detector'] = 0
                for wavelength in self.wavelength_list:
                    G.nodes[node]['laser'][wavelength] = []
                    G.nodes[node]['laser_capacity'][wavelength] = {}
                    G.nodes[node]['detector'][wavelength] = []
        return G

    def print_topology(self):
        print(f"Topology: {self.name}")
        for edge in self.physical_topology.edges(data=True):
            print(f"{edge[0]} —— {edge[1]}: {edge[2]['distance']} km")


# Example usage
if __name__ == '__main__':
    network = Network(map_name="Large", wavelength_list=[1], protocol="BB84", receiver="APD")
    network.print_topology()
    G = network.physical_topology
    # shortest_path = nx.dijkstra_path(G, source=25, target=23, weight='distance')
    # print(shortest_path)
