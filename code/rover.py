import argparse
from distutils.util import strtobool
import os
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist


def same_distance(a, b):
    if a == b:
        return 0
    else:
        return 1


def naive_loss(multi_unit, new_unit):
    # naive loss for alignment
    error = 0
    for unit in multi_unit:
        if unit != new_unit:
            error += 1
    return error / len(multi_unit)


numbers = "0123456789"

def check_contain_chinese(check_str):
    flag = False
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
        else:
            continue
    return flag

def load_hyp_dict(filename):
    # for each file, it contains decoded results like "uttid trans"
    with open(filename, "r", encoding="utf-8") as test_file:
        result = {}
        while True:
            hyp_line = test_file.readline()
            if not hyp_line:
                break
            hyp_array = hyp_line.strip().split()
            if len(hyp_array) >= 2:
                result[hyp_array[0]] = " ".join(hyp_array[1:])
            else:
                result[hyp_array[0]] = ""
            # hyp_line = hyp_line.strip()
            # for i in range(len(hyp_line)):
            #     if hyp_line[i] == "(" and "(" not in hyp_line[i + 1 :]:
            #         hyp_place = i
            # key = hyp_line[hyp_place + 1 : -1]

            # result[key] = hyp_line[:hyp_place]
    return result


class Distance(object):
    def __init__(self, delimiter, loss_fn, space2str=True):
        self.loss_fn = loss_fn
        self.delimiter = delimiter
        self.space2str = space2str
        self.INS, self.DEL, self.FOR = 1, 2, 3

    def multi_loss(self, multi_unit, new_unit):
        if self.loss_fn is not None:
            return self.loss_fn(multi_unit, new_unit)
        error = 0
        for unit in multi_unit:
            if unit != new_unit:
                error += 1

        return error / len(multi_unit)

    def alignment(self, hyp, ref, norm=False, return_er=False):
        assert type(ref) == str and type(hyp) == str

        if self.delimiter != "":
            ref = ref.split(self.delimiter)
            hyp = hyp.split(self.delimiter)

        ref_len = len(ref)
        hyp_len = len(hyp)

        workspace = np.zeros((len(hyp) + 1, len(ref) + 1))
        paths = np.zeros((len(hyp) + 1, len(ref) + 1))

        for i in range(1, len(hyp) + 1):
            workspace[i][0] = i
            paths[i][0] = self.INS
        for i in range(1, len(ref) + 1):
            workspace[0][i] = i
            paths[0][i] = self.DEL

        for i in range(1, len(hyp) + 1):
            for j in range(1, len(ref) + 1):
                if hyp[i - 1] == ref[j - 1]:
                    workspace[i][j] = workspace[i - 1][j - 1]
                    paths[i][j] = self.FOR
                else:
                    ins_ = workspace[i - 1][j] + 1
                    del_ = workspace[i][j - 1] + 1
                    sub_ = workspace[i - 1][j - 1] + self.loss_fn(
                        hyp[i - 1], ref[j - 1]
                    )
                    if ins_ < min(del_, sub_):
                        workspace[i][j] = ins_
                        paths[i][j] = self.INS
                    elif del_ < min(ins_, sub_):
                        workspace[i][j] = del_
                        paths[i][j] = self.DEL
                    else:
                        workspace[i][j] = sub_
                        paths[i][j] = self.FOR

        i = len(hyp)
        j = len(ref)
        aligned_hyp = []
        aligned_ref = []
        while i >= 0 and j >= 0:
            if paths[i][j] == self.FOR:
                aligned_hyp.append(hyp[i - 1])
                aligned_ref.append(ref[j - 1])
                i = i - 1
                j = j - 1
            elif paths[i][j] == self.DEL:
                aligned_hyp.append("*")
                aligned_ref.append(ref[j - 1])
                j = j - 1
            elif paths[i][j] == self.INS:
                aligned_hyp.append(hyp[i - 1])
                aligned_ref.append("*")
                i = i - 1
            if i == 0 and j == 0:
                i -= 1
                j -= 1

        aligned_hyp = aligned_hyp[::-1]
        aligned_ref = aligned_ref[::-1]

        if self.space2str:
            for i in range(len(aligned_hyp)):
                if aligned_hyp[i] == " ":
                    aligned_hyp[i] = "<space>"
            for i in range(len(aligned_ref)):
                if aligned_ref[i] == " ":
                    aligned_ref[i] = "<space>"

        # print("CER: {}".format(workspace[-1][-1]))
        if return_er:
            return workspace[-1][-1] / len(ref)
        return aligned_hyp, aligned_ref

    def skeleton_align(self, skeleton, align):
        if type(skeleton) == str:
            # detect raw text, format skeleton
            new_skeleton = []
            if self.delimiter != "":
                skeleton = skeleton.split(self.delimiter)
            for unit in skeleton:
                new_skeleton.append([unit])
            skeleton = new_skeleton

        if self.delimiter != "":
            align = align.split(self.delimiter)

        skeleton_len = len(skeleton)
        align_len = len(align)

        workspace = np.zeros((align_len + 1, skeleton_len + 1))
        paths = np.zeros((align_len + 1, skeleton_len + 1))

        for i in range(1, align_len + 1):
            workspace[i][0] = i
            paths[i][0] = self.INS
        for i in range(1, skeleton_len + 1):
            workspace[0][i] = i
            paths[0][i] = self.DEL

        for i in range(1, align_len + 1):
            for j in range(1, skeleton_len + 1):
                ins_ = workspace[i - 1][j] + self.multi_loss(
                    ["*"] * len(skeleton[0]), align[i - 1]
                )
                del_ = workspace[i][j - 1] + self.multi_loss(skeleton[j - 1], "*")
                sub_ = workspace[i - 1][j - 1] + self.multi_loss(
                    skeleton[j - 1], align[i - 1]
                )
                if ins_ < min(del_, sub_):
                    workspace[i][j] = ins_
                    paths[i][j] = self.INS
                elif del_ < min(ins_, sub_):
                    workspace[i][j] = del_
                    paths[i][j] = self.DEL
                else:
                    workspace[i][j] = sub_
                    paths[i][j] = self.FOR

        i = align_len
        j = skeleton_len
        new_skeleton = []
        while i >= 0 and j >= 0:
            if paths[i][j] == self.FOR:
                new_skeleton.append(skeleton[j - 1] + [align[i - 1]])
                i = i - 1
                j = j - 1
            elif paths[i][j] == self.DEL:
                new_skeleton.append(skeleton[j - 1] + ["*"])
                j = j - 1
            elif paths[i][j] == self.INS:
                new_skeleton.append(["*"] * len(skeleton[0]) + [align[i - 1]])
                i = i - 1
            if i == 0 and j == 0:
                i -= 1
                j -= 1

        new_skeleton = new_skeleton[::-1]

        return new_skeleton

    def process_skeleton_count(self, skeleton, weight=None, word=False, en_weight_factor=1.0):
        if weight is None:
            weight = [1 / len(skeleton[0])] * len(skeleton[0])
        result = []
        # import pdb;pdb.set_trace()
        for multi_unit in skeleton:
            temp_dict = {}
            for i in range(len(multi_unit)):
                if multi_unit[i] == "⁇":
                    # import pdb;pdb.set_trace()
                    temp_dict[multi_unit[i]] = temp_dict.get(multi_unit[i], 0)
                elif not check_contain_chinese(multi_unit[i]):
                    temp_dict[multi_unit[i]] = temp_dict.get(multi_unit[i], 0) + en_weight_factor * weight[i]
                else:
                    temp_dict[multi_unit[i]] = temp_dict.get(multi_unit[i], 0) + weight[i]
            result.append(max(temp_dict.items(), key=(lambda item: item[1]))[0])

        return " ".join(result) if word else "".join(result)

    def save_skeleton(self, skeleton):
        num_hyp = len(skeleton[0])
        aligned_string = []
        for i in range(num_hyp):
            aligned_string.append("")
        for multi_unit in skeleton:
            for i in range(num_hyp):
                aligned_string[i] += multi_unit[i]
        return "@@@".join(aligned_string)


def combine(hyp_list, loss, word=False, weight=None, en_weight_factor=1.0):
    # add weight for different weight of voting process
    if word:
        delimiter = " "
    else:
        delimiter = ""
    dist_char = Distance(delimiter=delimiter, loss_fn=loss, space2str=True)

    assert len(hyp_list) > 1
    skeleton = hyp_list[0]

    for align in hyp_list[1:]:
        skeleton = dist_char.skeleton_align(skeleton, align)

    return dist_char.process_skeleton_count(skeleton, weight=weight, word=word, en_weight_factor=en_weight_factor)


def generate_alignment(hyp_list, loss, word=False):
    if word:
        delimiter = " "
    else:
        delimiter = ""
    dist_char = Distance(delimiter=delimiter, loss_fn=loss, space2str=True)

    assert len(hyp_list) > 1
    skeleton = hyp_list[0]

    for align in hyp_list[1:]:
        skeleton = dist_char.skeleton_align(skeleton, align)

    return dist_char.save_skeleton(skeleton)


def check_keys(key, source_list):
    for source in source_list:
        if key not in source.keys():
            return False
    return True


def test():
    test = ["abcd", "bcdeeeee", "bfeeeee"]
    dist_char = Distance(delimiter="", loss_fn=same_distance, space2str=True)
    print(combine(test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources", type=str, help="a list of source hyp files, separate with ,"
    )
    parser.add_argument("--result", type=str, default="rover-result.txt")
    parser.add_argument("--word", type=strtobool, default=False)
    parser.add_argument(
        "--distance_type",
        type=str,
        default="cosine",
        help="cosine, euclidean, seuclidean",
    )
    parser.add_argument(
        "--save_align",
        type=strtobool,
        default=False,
        help="save alignment file for analysis",
    )
    parser.add_argument(
        "--en_weight_factor",
        type=float,
        default=1.0,
        help="english weight factor for vote",
    )

    args = parser.parse_args()

    print("Combin hyps: {}".format(args.sources))

    loss = naive_loss

    source_list = args.sources.split(",")

    source_dicts = []

    for i in range(len(source_list)):
        source_dicts.append(load_hyp_dict(source_list[i]))

    result_file = open(args.result, "w", encoding="utf-8")

    for key in source_dicts[0].keys():
        if not check_keys(key, source_dicts):
            print("missing key {} in source".format(key))
        hyp_list = []
        for source in source_dicts:
            if key in source.keys() and source[key] != "":
                hyp_list.append(source[key])

        if len(hyp_list) == 0:
            print("all hyps are empty for utt {}".format(key))
            result_file.write("{} {}\n".format(key, final))
            continue
        if len(hyp_list) == 1:
            print("only have one unempty hyp for utt {}".format(key))
            result_file.write("{} {}\n".format(key, final))
            continue

        if args.save_align:
            final = generate_alignment(hyp_list, loss, args.word)
            result_file.write("{} {}\n".format(key, final))
        else:
            final = combine(hyp_list, loss, word=args.word, weight=None, en_weight_factor=args.en_weight_factor)
            final = final.replace("*", "")
            result_file.write("{} {}\n".format(key, final))
        # print(key, final)
    result_file.close()
