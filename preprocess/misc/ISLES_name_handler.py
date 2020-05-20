import os
import re


class ISLESNameHandler:

    @staticmethod
    def parse_name(root_path, filename):
        """
        from each file name, return the label for such file for storage.
        :return:
        """
        label_names = []
        match_id_result = re.search(r"(OT|TTP|Tmax|rCBV|rCBF|MTT|ADC).+.nii.gz", filename)
        if match_id_result is None:
            return  os.path.basename(root_path), None
        matches = match_id_result.groups()

        if matches[0] == "OT":
            label_names.append("label")
        else:
            label_names.append(f"data/{matches[0]}")

        match_path_result = re.search(r"training_(\d+)", root_path)

        # parent_folder = os.path.basename(root_path)
        if match_path_result is None:
            parent_folder = os.path.basename(root_path)
        else:
            parent_folder = match_path_result.groups()[0]
            label_names.insert(0,match_path_result.groups()[0])

        return parent_folder, "/".join(label_names)
