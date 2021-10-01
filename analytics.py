import numpy as np
import pandas as pd
import glob
import json

results_files = glob.glob("outputs/reduced_*")
print(results_files)

# instances counter
dataframe = pd.read_csv("databases/AG/Aritmética/TreinamentoDesbalanceadoAritPreprocessada.csv")
num_instances = len(dataframe.index)
results = []
for result_file in results_files:
    results.append(json.load(open(result_file)))

counter_list = []
length_counter = []
reduction_ratios = []
reduction_ratios_per_test = []
reduction_ratios_per_cv = []
reduction_ratios_per_cv_it = []
for result in results:
    ratios_cv = []
    for key in result:
        if key.startswith('i'):
            selected_instances_in_cv = result[key]["selected_indices"]
            red_it_ratios = []
            for selected_instances in selected_instances_in_cv:
                red_it_ratios.append(len(selected_instances) / num_instances)
                reduction_ratios.append(len(selected_instances) / num_instances)
                counter_array = np.zeros(num_instances)
                counter_array[selected_instances] += 1
                counter_list.append(counter_array)

            ratios_cv.append(np.mean(red_it_ratios))
            reduction_ratios_per_cv.append(np.mean(red_it_ratios))

    reduction_ratios_per_test.append(np.mean(ratios_cv))

print()
print("---===Reduction Rates===---")
print("*All cross validation iterations*")
print(f"Mean: {np.mean(reduction_ratios)}")
print(f"Median: {np.median(reduction_ratios)}")
print(f"Min: {np.min(reduction_ratios)}")
print(f"Max: {np.max(reduction_ratios)}")
print()
print("*Per iteration of complete cross validation*")
print(f"Mean: {np.mean(reduction_ratios_per_cv)}")
print(f"Median: {np.median(reduction_ratios_per_cv)}")
print(f"Min: {np.min(reduction_ratios_per_cv)}")
print(f"Max: {np.max(reduction_ratios_per_cv)}")
print()
print("*Per Test*")
print(f"Mean: {np.mean(reduction_ratios_per_cv)}")
print(f"Median: {np.median(reduction_ratios_per_cv)}")
print(f"Min: {np.min(reduction_ratios_per_cv)}")
print(f"Max: {np.max(reduction_ratios_per_cv)}")
print()
print("---===Class analsys===---")


counter_list.append(np.zeros(num_instances))
counter_list = np.array(counter_list)
counter_row = np.zeros(num_instances)
for i in range(num_instances):
    x = counter_list[:, i]
    counter_row[i] = np.sum(counter_list[:, i])

counter_list[-1] = counter_row


counter_row_per_instance = \
    np.concatenate((np.arange(num_instances).reshape(-1, 1), counter_row.reshape(-1, 1)), axis=1)

sorted_counter_row = \
    counter_row_per_instance[counter_row_per_instance[:, 1].argsort()][::-1]

instances_df = pd.DataFrame(counter_list, columns=["i" + str(i) for i in range(num_instances)])
instances_df = instances_df.astype("int32")
instances_df.to_csv("outputs/instancias_selecionadas.csv", index=False)
sorted_counter_df = pd.DataFrame(sorted_counter_row, columns=["instance", "num_selected"])
sorted_counter_df = sorted_counter_df.astype('int32')
print(sorted_counter_df)

# print(instances_df)





