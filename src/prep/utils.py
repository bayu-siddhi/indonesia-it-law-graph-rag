"""Some utility functions for reading, writing, and manipulating regulation data."""

import json
from typing import Any, Dict, List, Union

import pandas as pd
from tqdm import tqdm


def read_json(input_path: str) -> Any:
    """
    Reads JSON data from the file specified by the input path.

    Args:
        input_path (str): The path to the JSON file.

    Returns:
        result (Any): The JSON data loaded from the file.
    """
    if not input_path.endswith(".json"):
        input_path = input_path + ".json"
    with open(input_path, "r", encoding="utf-8") as input_file:
        json_data = json.load(input_file)
    return json_data


def list_of_dict_to_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Writes a list of dictionaries to a JSON file, adding the .json extension if 
    needed.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to write.
        output_path (str): The path to the output JSON file.
    
    Returns:
        None
    """
    if not output_path.endswith(".json"):
        output_path = output_path + ".json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def list_of_dict_to_excel(
    data: List[Dict[str, Any]], output_path: str, sheet_name: str
) -> None:
    """
    Writes a list of dictionaries to an Excel file (creates or appends to the 
    specified sheet).

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to write to Excel.
        output_path (str): The path to the output Excel file.
        sheet_name (str): The name of the sheet to write to in the Excel file.
    
    Returns:
        None
    """
    if not output_path.endswith(".xlsx"):
        output_path = output_path + ".xlsx"

    df = pd.DataFrame(data)

    try:
        with pd.ExcelWriter(
            output_path, engine="openpyxl", mode="a", if_sheet_exists="replace"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    except FileNotFoundError:
        with pd.ExcelWriter(output_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def load_excel_selected_regulations(
    file_path: str, sheet_name: str, url_type: str = "url_1", url_only: bool = True
) -> Union[List[str], List[Dict[str, Any]]]:
    """
    Loads selected regulations from an Excel file based on specified criteria.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str): The name of the sheet to load from.
        url_type (str, optional): The column name containing the URL. Defaults to 
            "url_1".
        url_only (bool, optional): If True, returns only the list of URLs; otherwise, 
            returns a list of dictionaries containing "name" and "url". Defaults to 
            True.

    Returns:
        result (Union[List[str], List[Dict[str, Any]]]): A list of URLs or a list of 
            dictionaries with "name" and "url" keys.
    """

    # Read All Regulation with "used" == "YA"
    selected_regulations = pd.read_excel(file_path, sheet_name=sheet_name)
    selected_regulations = selected_regulations.loc[
        selected_regulations["used"] == "YA"
    ].copy()

    if url_only:
        return selected_regulations[url_type].tolist()
    selected_regulations = selected_regulations.loc[
        selected_regulations[url_type].notna()
    ].copy()
    selected_regulations = selected_regulations.loc[:, ["name", url_type]].copy()
    selected_regulations.rename(
        columns={"name": "name", url_type: "url"}, inplace=True
    )
    return selected_regulations.to_dict(orient="records")


def modify_status_json_regulation(input_json_file: str, verbose: bool = True) -> None:
    """
    Modifies the 'status' field in a JSON file by replacing URLs with their corresponding 
    IDs.

    Args:
        input_json_file (str): The path to the input JSON file.
        verbose (bool, optional): Whether to print progress information. Defaults to 
            True.
    
    Returns:
        None
    """
    json_data = read_json(input_json_file)

    mapping_url_id = {}
    for regulation in json_data:
        mapping_url_id[regulation["url"]] = regulation["id"]

    for index, regulation in tqdm(
        iterable=enumerate(json_data),
        desc="Modify regulation metadata",
        disable=not verbose,
    ):
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
