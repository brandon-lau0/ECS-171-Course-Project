import sys
import json
import math

def main():

  if len(sys.argv) != 4:
    print("Usage Ex. py splitparams.py params.json <prefix of new file> <num splits>")
    sys.exit()

  param_file = sys.argv[1]
  prefix = sys.argv[2]
  num_splits = sys.argv[3]

  with open(param_file, 'r') as fin:
    l_params = json.load(fin)

    file_length = math.ceil(len(l_params)/int(num_splits))
    
    new_params = []
    count = 0
    for i, dictionary in enumerate(l_params):
      if len(new_params) < file_length:
        new_params.append(dictionary)
      else:
        with open(prefix + str(count) + '.json', 'w') as fout:
          json.dump(new_params, fout)
        count += 1
        new_params = []
        new_params.append(dictionary)

    if len(new_params):
        with open(prefix + str(count) + '.json', 'w') as fout:
          json.dump(new_params, fout)
        

if __name__ == '__main__':
  main()