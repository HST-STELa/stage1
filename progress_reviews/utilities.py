from xml.etree import ElementTree as ET


def parse_visit_labels_from_xml_status(xml_path):
    # parse the xml visit status export from STScI
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Iterate over each visit element in the visit status, find the appropriate row, and update dates
    labels = []
    for visit in root.findall('visit'):
        visit_label = visit.attrib.get('visit')
        labels.append(visit_label)

    return labels
