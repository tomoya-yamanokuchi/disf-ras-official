import xml.etree.ElementTree as ET


def indent_tree(elem, level=0):
    """
    XMLツリーに改行とインデントを追加する関数。
    Args:
        elem: XMLのルート要素
        level: 現在のインデントレベル
    """
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_tree(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
