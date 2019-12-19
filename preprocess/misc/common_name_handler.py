import os
import re


class CommonNameHandler:

    @staticmethod
    def parse_name(root_path, filename):
        """
        from each file name, return the label for such file for storage.
        :return:
        """
        label_names = []
        match_id_result = re.search(r"([^_]+)_(frame(\d+))?", filename)

        matches = match_id_result.groups()
        if matches[0] is not None:
            label_names.append(matches[0])
        if matches[-1] is not None:
            label_names.append(matches[-1])
        match_gt_result = re.search(r"_gt\.nii\.gz", filename)
        match_data_result = re.search(r"_frame(\d+)\.nii\.gz", filename)
        if match_gt_result is not None:
            label_names.append("label")
        elif match_data_result is not None:
            label_names.append("data")
        else:
            label_names.append("data_without_label")

        parent_folder = os.path.basename(root_path)

        return parent_folder, "/".join(label_names)