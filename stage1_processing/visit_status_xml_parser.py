import re
from datetime import datetime

from astropy import table
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


def parse_planwindow_date(text):
    """ Extracts the left-most date from a planWindow string. Expected input example: "Mar 31, 2025 - Apr 1, 2025 (2025.090 - 2025.091)" This function takes the first date (e.g. "Mar 31, 2025") and returns a datetime object. """
    if text.startswith('Not'):
        return None
    else:
        date_part = text.split(" - ")[0].strip() # Parse the date; expected format e.g. "Mar 31, 2025"
        parsed_date = datetime.strptime(date_part, "%b %d, %Y")
        return parsed_date


def load_visit_status_xml_as_table(xml_path):
    # Iterate over each visit element in the visit status, find the appropriate row, and update dates
    # this can't be done with a standard table join given a row can be associated with multiple visits if there are redos
    tree = ET.parse(xml_path)
    root = tree.getroot()

    rows = []
    for visit in root.findall('visit'):
        row = {}
        visit_label = visit.attrib.get('visit')
        row['visit'] = visit_label

        target_items = visit.findall('target')
        targets = []
        for target in target_items:
            nonprimary = re.findall('WAVE|OFFSET', target.text)
            if not nonprimary:
                targets.append(target.text)
        assert len(targets) == 1
        row['target'] = targets[0]

        # Get the status text (if available)
        status_elem = visit.find('status')
        status = status_elem.text.strip() if status_elem is not None else ""

        row['status'] = status

        # Get all planWindow elements (if any)
        plan_windows = visit.findall('planWindow')

        start_time_elem = visit.find('startTime')
        if start_time_elem is not None and start_time_elem.text:
            actual_obs, = re.findall(r'\w{3} \d+, \d{4}', start_time_elem.text)
            row['obsdate'] = datetime.strptime(actual_obs, "%b %d, %Y")

        dates = []
        for pw in plan_windows:
            if pw.text:
                dt = parse_planwindow_date(pw.text)
                if dt:
                    dates.append(dt)
        if dates:
            row['next'] = min(dates)

        rows.append(row)

    status_tbl = table.Table(rows=rows)
    status_tbl = status_tbl['target visit status obsdate next'.split()]
    return status_tbl