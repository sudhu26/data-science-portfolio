
import json
import pandas as pd

path = '/usr/local/spark-2.4.3/spark-2.4.3-bin-hadoop2.7/examples/src/main/resources/people.json' 

with open(path, 'r') as handle: 
    data = [json.loads(line) for line in handle] 

print(data)

df = pd.DataFrame(data)

display(df)
