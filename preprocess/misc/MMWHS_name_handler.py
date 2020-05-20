import os
import re


class MMWHSNameHandler:

    @staticmethod
    def parse_name(root_path, filename):
        """
        from each file name, return the label for such file for storage.
        :return:
        """
        label_names = []
        match_id_result = re.search(r"ct_train_(\d+)_(\S+).nii.gz", filename)
        if match_id_result is None:
            return  os.path.basename(root_path), None
        matches = match_id_result.groups()
        label_names.append(matches[0])
        if matches[1] == "image":
            label_names.append("data")
        else:
            label_names.append("label")

        parent_folder = os.path.basename(root_path)

        return parent_folder, "/".join(label_names)
