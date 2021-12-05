from svgpathtools import svg2paths2, parse_path, wsvg
from os import listdir
from os.path import isfile, join


def flatten_transform(file_path, new_file_path):
    paths, attributes, svg_attributes = svg2paths2(file_path)
    paths = []
    new_attributes = []
    for attr in attributes:
        translations = eval(attr['transform'].split('translate')[1])
        new_path = parse_path(attr['d']).translated(complex(translations[0], translations[1]))
        paths.append(new_path)
        new_attr = {'d': new_path.d(), 'fill': attr['fill']}
        new_attributes.append(new_attr)
    wsvg(paths=paths, attributes=new_attributes, filename=new_file_path)


if __name__ == "__main__":
    svgs = [f for f in listdir("../temp/svgs") if isfile(join("../temp/svgs", f))]
    for f in svgs:
        print(f)
        flatten_transform("../temp/svgs/{}".format(f), "../temp/transformed_svgs/{}".format(f))
