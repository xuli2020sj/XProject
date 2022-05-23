import re
import pandas as pd

input_file = r'../log/log5_18.txt'
output_file = r'../data/log5_18_detector_front.csv'

with open(input_file, 'r') as f:
    data = f.readlines()

pos_pattern = '[(].*[)]'
eff_pattern = 'Detection efficiency:.*'
ene_pattern = 'Source energy: .*'
pos_pattern_compiled = re.compile(pos_pattern)
eff_pattern_compiled = re.compile(eff_pattern)
ene_pattern_compiled = re.compile(ene_pattern)
pos_list = []
eff_list = []
ene_list = []

for line in data:
    pos_res = pos_pattern_compiled.findall(line)
    eff_res = eff_pattern_compiled.findall(line)
    ene_res = ene_pattern_compiled.findall(line)

    if pos_res:
        pos = pos_res[0][1:-1].split(',')
        pos_list.append([float(pos[0]), float(pos[1]), float(pos[2])])
    if eff_res:
        eff_list.append(float(eff_res[0][21:]))
    if ene_res:
        ene_list.append(float(ene_res[0][14:]))
df_pos = pd.DataFrame(pos_list, columns=['xPos', 'yPos','zPos'], dtype=float)
df_eff = pd.DataFrame(eff_list, columns=['Efficiency'], dtype=float)
df_ene = pd.DataFrame(ene_list, columns=['Energy'], dtype=float)

df = pd.concat([df_pos, df_ene, df_eff], axis=1)
df.to_csv(output_file)


