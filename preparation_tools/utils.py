import json
import tqdm
import pandas as pd


# https://stackoverflow.com/questions/20199126/reading-json-from-a-file
# https://stackoverflow.com/questions/27092833/unicodeencodeerror-charmap-codec-cant-encode-characters
def read_json(input_path: str):
    if not input_path.endswith(".json"):
        input_path = input_path + ".json"
    with open(input_path, encoding="utf-8") as input_file:
        json_data = json.load(input_file)
    return json_data


def list_of_dict_to_json(data: list[dict], output_path: str) -> None:
    if not output_path.endswith(".json"):
        output_path = output_path + ".json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def list_of_dict_to_excel(data: list[dict], output_path: str, sheet_name: str) -> None:
    if not output_path.endswith(".xlsx"):
        output_path = output_path + ".xlsx"
    df = pd.DataFrame(data)
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def load_excel_selected_regulations(
        file_path: str,
        sheet_name: str,
        url_type: str = "url_1",
        url_only: bool = True ) -> list[str] | list[dict]:

    # Read All Regulation with "used" == 1
    selected_regulations = pd.read_excel(file_path, sheet_name=sheet_name)
    selected_regulations = selected_regulations.loc[selected_regulations["used"] == 1].copy()

    if url_only:
        return selected_regulations[url_type].tolist()
    else:
        selected_regulations = selected_regulations.loc[selected_regulations[url_type].notna()].copy()
        selected_regulations = selected_regulations.loc[:, ["name", url_type]].copy()
        selected_regulations.rename(columns={"name": "name", url_type: "url"}, inplace=True)
        return selected_regulations.to_dict(orient="records")


def modify_status_json_regulation(input_json_file: str, verbose: bool = True) -> None:
    json_data = read_json(input_json_file)

    mapping_url_id = {}
    for regulation in json_data:
        mapping_url_id[regulation["url"]] = regulation["id"]

    for index, regulation in tqdm.tqdm(iterable=enumerate(json_data), desc="Modify regulation metadata", disable=not verbose):
        for status, list_urls in regulation["status"].items():
            temp_list = []
            for url in list_urls:
                mapped_value = mapping_url_id.get(url, url)
                temp_list.append(mapped_value)
            json_data[index]["status"][status] = temp_list

    if input_json_file.endswith(".json"):
        output_json_file = input_json_file[:-5] + "_modified.json"
    else:
        output_json_file = input_json_file + "_modified.json"

    list_of_dict_to_json(json_data, output_json_file)
    
    if verbose:
        print(f"Successfully modified JSON data to {output_json_file}")
