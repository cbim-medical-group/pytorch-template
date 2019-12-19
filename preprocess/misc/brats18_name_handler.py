import os
import re


class Brats18NameHandler:

    @staticmethod
    def parse_name(root_path, filename):
        """
        from each file name, return the label for such file for storage.
        :return:
        """
        label_names = []
        match_id_result = re.search(r"([^_]+).nii.gz", filename)

        matches = match_id_result.groups()
        if matches[0] is not None:
            if matches[0] == "seg":
                label_names.append("label")
            else:
                label_names.append("data")
                label_names.append(matches[0])
        parent_folder = os.path.basename(root_path)
        return parent_folder, "/".join(label_names)
