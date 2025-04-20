result = {'traffic': 1050000, 'total_avg_power': 80.44272953414745, 'avg_spectrum_occupied': 0.06095238095238094, 'avg_component_power': {'source': 0.4047942107643599, 'detector': 3.0359565807326994, 'other': 77.0019787426504, 'ice_box': 0.0}}

target = result['total_avg_power']
component = result['avg_component_power']
sum = 0
for key, value in component.items():
    sum += value
for key, value in component.items():
    result['avg_component_power'][key] = target * value / sum

sum = 0
for key, value in result['avg_component_power'].items():
    sum += value


print(result)
print(result['total_avg_power'], sum)