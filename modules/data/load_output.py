import json
import glob
import json
import glob

def load_results_from_term(path, term, algo_names):
    results_data = {}
    for algo_name in algo_names:
        path_full = glob.glob(path + "full_" + term + "_" + algo_name + "*")[0]
        path_reduced = glob.glob(path + "reduced_" + term + "_" + algo_name + "*")[0]
        path_resampled = glob.glob(path + "resampled_" + term + "_" + algo_name + "*")[0]
        path_improved = glob.glob(path + "improved_" + term + "_" + algo_name + "*")[0]
        print(f"Full {algo_name}:  {path_full}")
        print(f"Reduced {algo_name}:  {path_reduced}")
        print(f"Resampled {algo_name}:  {path_resampled}")
        print(f"Improved {algo_name}:  {path_improved}")
        results_data[algo_name] = {
            "full": json.load(open(path_full)),
            "reduced": json.load(open(path_reduced)),
            "resampled": json.load(open(path_resampled)),
            "improved": json.load(open(path_improved)) 
        }
    
    return results_data



def load_output(path_to_output_dir, algo_names):
    arit = load_results_from_term(path_to_output_dir, "arit", algo_names)
    print("-----------------------")
    esc = load_results_from_term(path_to_output_dir, "esc", algo_names)
    print("---------------------------")
    leit = load_results_from_term(path_to_output_dir, "leit", algo_names)

    return arit, esc, leit