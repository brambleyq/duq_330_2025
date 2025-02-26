from dataclasses import dataclass
import xml.etree.ElementTree as ET

@dataclass
class Assignee():
    name: str
    address: str

    def is_valid(self):
        if self.name is None or len(self.name) == 0:
            return False
        if self.address is None:
            return False
        return True

@dataclass
class Assignor():
    forename: str
    surname: str

    def is_valid(self):
        if (self.forename is None or 
            len(self.forename) == 0 or 
            self.surname is None 
            or len(self.surname) == 0
        ):
            return False
        return True

# @ is "decorator" 
@dataclass
class Patent():
    assignors: list[Assignor]
    assignees: list[Assignee]
    name: str
    document_ids: list[str]

    def is_valid(self):
        if self.assignors is None or len(self.assignors) == 0:
            return False
        if self.name is None or len(self.name) == 0:
            return False
        
        for assignor in self.assignors:
            if not assignor.is_valid():
                return False
        for assignee in self.assignees:
            if not assignee.is_valid():
                return False
            
        return True

def parse_address(company:ET.Element) -> str|None:
    """takes an assignee xml element and returns 
    the adress (formated to the best of my ability)

    Args:
        company (ET.Element): an xml element of <patent-assignee>

    Returns:
        str|None: an adress needs all city, state, and postal code and
        atleast 1 adress line
    """
    address = ''
    line = 0
    while True:
        line += 1
        xml_line = company.find(f'address-{line}')
        if xml_line is None:
            break
        address += xml_line.text + '\n'
    xml_city = company.find('city')
    xml_state = company.find('state')
    xml_postcode = company.find('postcode')
    if xml_city is None:
        return None
    if xml_state is None:
        return None
    if xml_postcode is None:
        return None
    city = xml_city.text
    state = xml_state.text
    postcode = xml_postcode.text
    address += f'{city} {state} {postcode}' 
    return address



def parse_assignment(element:ET.Element) -> Patent|None:
    """parse a specific patent assigment from an element

    Args:
        element (ET.Element): an xtml element
    """
    assignors = element.find('patent-assignors')
    assinees = element.find('patent-assignees')
    records = element.find('patent-properties')
    if assignors is None:
        return None
    if records is None:
        return None
    if assinees is None:
        return None

    people = []
    companies = []
    for company in assinees.findall('patent-assignee'):
        xml_name = company.find('name')
        if xml_name is None:
            continue
        name = xml_name.text
        address = parse_address(company)
        companies.append(Assignee(name=name,address=address))


    for person in assignors.findall('patent-assignor'):
        try:
            name = person.find('name')
            if name is None:
                # no name
                continue
            name_parts = name.text.split(', ')
            surname = name_parts[0]
            forename = name_parts[1]
            people.append(Assignor(forename=forename,surname=surname))
        except IndexError:
            pass
        
    
    doc_ids = []
    patent_name = None
    for record in records.findall('patent-property'):
        for title in record.findall('invention-title'):
            if patent_name is None:
                patent_name = title.text
        for doc in record.findall('document-id'):
            for num in doc.findall('doc-number'):
                doc_ids.append(num.text)
    return Patent(document_ids=doc_ids,name=patent_name,assignors=people,assignees=companies)


def parse_patents(path:str) -> list[Patent]:
    """parses patents from a xtml file returning a list of patents

    Args:
        path (str): path to xtml

    Returns:
        list[Patent]: list of patents
    """
    tree = ET.parse(path)
    root = tree.getroot()
    
    out = []
    assignments = root.find('patent-assignments')
    if assignments is None:
        return []
    for el in assignments.findall('patent-assignment'):
        out.append(parse_assignment(el))
    
    return [patent for patent in out if patent is not None and patent.is_valid()]

if __name__ == '__main__':
    all_patents = parse_patents('data/ad20250218.xml')
    print('debug')