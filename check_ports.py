#!/usr/bin/python3

cmd = "netstat -antu | grep LISTEN | awk '{print $4,$6}' | column -t | grep '127.0.0.1'"
strg = subprocess.check_output(cmd, shell=True).decode("utf-8").split('\n')
output = [ el.split('   ') for el in strg ]
output = [ [el[0].strip(),el[1].strip()] for el in output if len(el) > 1 ]
ports = pd.DataFrame([ [el[0].strip().split(':')[-1],el[1].strip()] for el in output if len(el) > 1 ],columns=['port','state'])
# nc -zv 127.0.0.1 6463
PORT = str(ports[ports['state']=='LISTEN'].sample()['port'].values[0].split(':')[-1])

