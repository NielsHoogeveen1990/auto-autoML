def prettyprint(d, indent=0):
   for key, value in d.items():
      print('\t' * indent + str(key))
      if isinstance(value, dict):
         prettyprint(value, indent+1)
      else:
         print('\t' * (indent+1) + str(value))


def get_best_automl_package(result_dict):
    highest_score = 0
    for k, v in result_dict.items():
        if v['cross_validation_mean'] > highest_score:
            highest_score = v['cross_validation_mean']
            key = k

    return key, result_dict[key]['parameters']