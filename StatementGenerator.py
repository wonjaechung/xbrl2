# StatementGenerator.py

import collections
import pandas as pd
from datetime import datetime
import re

class StatementGenerator:
    """
    Generates traditional financial statements (e.g., Income Statement)
    in a structured Markdown format based on parsed XBRL data.
    """
    def __init__(self, parser, all_documents):
        self.parser = parser
        self.all_documents = all_documents
        self.taxonomy = parser.taxonomy_data
        self.presentation_links = self._get_presentation_links(parser)
        self.role_definitions = {role['roleURI']: f"{role['Name_EN']} {role['ID']}" for role in parser.role_definitions}
        self.child_map = self._build_child_map()

    def _get_presentation_links(self, parser):
        """Extracts and organizes presentation links from the parser's XML data."""
        links = collections.defaultdict(list)
        for concept_id, data in self.taxonomy.items():
            if 'relationships' in data and 'presentation_parent' in data['relationships']:
                for rel in data['relationships']['presentation_parent']:
                    parent_id = rel.get('parent')
                    role_uri = rel.get('roleURI')
                    order = rel.get('order', 1.0)
                    if parent_id and role_uri:
                        links[role_uri].append({'parent': parent_id, 'child': concept_id, 'order': order})
        # Sort links by order
        for role_uri in links:
            links[role_uri].sort(key=lambda x: x.get('order', 1.0))
        return links

    def _build_child_map(self):
        child_map = collections.defaultdict(list)
        for role_uri, links in self.presentation_links.items():
            for link in links:
                child_map[(role_uri, link['parent'])].append(link['child'])
        return child_map

    def _get_label(self, concept_id, lang='ko'):
        """Gets the best available label for a concept."""
        return self.parser._get_label_for_concept(concept_id, lang)

    def _get_fact_value(self, concept_id, period_end_date, dimensions):
        """
        Retrieves a numerical fact by parsing it from the generated concept document.
        This is more reliable than querying the complex parser object.
        """
        doc_content = self.all_documents.get(concept_id)
        if not doc_content:
            return None

        # Recreate the exact context ID string that DocumentGenerator would have created.
        if dimensions:
            context_id_string = " and ".join([f"{k}: {v}" for k, v in sorted(dimensions.items())])
            # Search for the exact ID string within the HTML comment
            search_pattern = f"<!-- Context (IDs): {context_id_string} -->"
        else:
            # For Primary Context, there's no ID comment, so we find the header directly
            search_pattern = "### **Context (Labels): Primary Context**"

        # Find the block of text for the correct context using simple string search
        if search_pattern in doc_content:
            # Find the start position
            start_pos = doc_content.find(search_pattern)
            # Find the end position (next context or accounting references)
            end_pos = doc_content.find("### **Context (Labels)", start_pos + len(search_pattern))
            if end_pos == -1:
                end_pos = doc_content.find("## Accounting Standard References", start_pos + len(search_pattern))
            if end_pos == -1:
                end_pos = len(doc_content)
            
            context_block = doc_content[start_pos:end_pos]
        else:
            return None

        # Within that block, find the correct period and extract the value
        # Look for period that ends with the specified date
        period_match = re.search(f"- \\*\\*Period: [^\\n]*to {re.escape(period_end_date)}[^\\n]*\\*\\*[^\\n]*\\n\\s*- \\*\\*Value\\*\\*: ([0-9,.-]+)", context_block)
        if period_match:
            value_str = period_match.group(1).replace(",", "")
            try:
                return float(value_str)
            except (ValueError, TypeError):
                return None
        return None

    def _format_bignum(self, num):
        """Formats a large number with commas."""
        if num is None:
            return ""
        if not isinstance(num, (int, float)):
            return ""
        return f"{num:,.0f}"

    def _generate_rows(self, role_uri, parent_concept, period_end_date, current_dimensions, level=0):
        rows = []
        children = self.child_map.get((role_uri, parent_concept), [])
        
        for child_concept in children:
            indent = "    " * level
            label = self._get_label(child_concept)
            
            # Keywords for elements that are purely structural and should never be printed.
            structural_keywords = ["Abstract", "Table", "Axis", "Domain", "Explanatory", "LineItems"]
            if any(keyword in child_concept for keyword in structural_keywords):
                # We don't print the row, but we must process its children to find the real data.
                # We pass the *same* level so indentation doesn't increase for purely structural hops.
                rows.extend(self._generate_rows(role_uri, child_concept, period_end_date, current_dimensions, level))
                continue

            # This concept is a dimensional member that defines a new context (e.g., a specific business segment)
            if "Member" in child_concept:
                new_dimensions = current_dimensions.copy()
                
                # Heuristically determine the dimension axis this member belongs to.
                axis = 'ifrs-full:SegmentConsolidationItemsAxis' # Default assumption
                if 'Geographical' in child_concept:
                    axis = 'ifrs-full:GeographicalAreasAxis'

                member_id = child_concept.replace('_', ':')
                new_dimensions[axis] = member_id
                
                # This is a header row, like "Bio-pharmaceuticals". Print it.
                rows.append(f'| {indent}{label} | |')

                # Now, recurse into its children, passing the NEW dimensional context.
                rows.extend(self._generate_rows(role_uri, child_concept, period_end_date, new_dimensions, level + 1))
                continue

            # If we've reached here, it's a standard, non-dimensional line item.
            # We fetch its value using the *current* dimensional context passed into the function.
            concept_id_for_search = child_concept.replace('_', ':')
            value = self._get_fact_value(concept_id_for_search, period_end_date, current_dimensions)
            
            if value is not None:
                amount_str = f"{value:,.0f}"
                rows.append(f'| {indent}{label} | {amount_str} |')

        return rows

    def generate_statement(self, role_uri, period_end_date, dimensions):
        """Generates a complete financial statement for a given role URI."""
        if role_uri not in self.role_definitions:
            return f"Role URI not found: {role_uri}"

        title = self.role_definitions[role_uri]
        md = f"# {title}\n"
        md += "| Account | Amount |\n"
        md += "|---|---|\n"
        
        # Find root concepts for this role
        root_concepts = [
            link['parent'] for link in self.presentation_links.get(role_uri, [])
            if not any(l['child'] == link['parent'] for l in self.presentation_links.get(role_uri, []))
        ]
        
        for root_concept in sorted(list(set(root_concepts))):
             # Initial call starts with the base dimensions
            rows = self._generate_rows(role_uri, root_concept, period_end_date, dimensions, level=0)
            md += "\n".join(rows)
            
        return md

    def generate_custom_operating_segments(self, role_uri, period_end_date):
        """
        Generates the 'Disclosure of Operating Segments' report with a hard-coded structure
        that matches the user's provided image exactly. This is less flexible but more reliable.
        """
        if role_uri not in self.role_definitions:
            return f"Role URI not found: {role_uri}"

        title = self.role_definitions[role_uri]
        md = f"# {title}\n"
        md += "| 계정 | 2025-01-01 ~ 2025-03-31 |\n"
        md += "|---|---|\n"

        # --- Define the structure of the report ---
        
        # Line items to report for each segment, using the most specific IDs available.
        line_items = {
            'ifrs-full:Revenue': '매출액',
            'ifrs-full:DepreciationExpense': '감가상각비',
            'entity00413046:AmortisationExpenseOfDisclosureOfOperatingSegmentsLineItemsOfDisclosureOfOperatingSegmentsTableOfItems': '무형자산상각비',
            'dart:OperatingIncomeLoss': '영업이익',
            'ifrs-full:NoncurrentAssetsOtherThanFinancialInstrumentsDeferredTaxAssetsPostemploymentBenefitAssetsAndRightsArisingUnderInsuranceContracts': '비유동자산'
        }

        # Segments (dimensions) to report on
        segments = {
            '연결재무제표': 'ifrs-full:OperatingSegmentsMember',
            '바이오의약품': 'entity00413046:BiopharmaceuticalMedicinesOfOperatingSegmentsMemberOfDisclosureOfOperatingSegmentsTableOfMember',
            '케미컬의약품': 'entity00413046:ChemicalMedicinesOfOperatingSegmentsMemberOfDisclosureOfOperatingSegmentsTableOfMember',
            '기타부문': 'ifrs-full:AllOtherSegmentsMember',
            '부문간 제거한 금액': 'ifrs-full:EliminationOfIntersegmentAmountsMember'
        }

        base_dimensions = {
            'ifrs-full:ConsolidatedAndSeparateFinancialStatementsAxis': 'ifrs-full:ConsolidatedMember'
        }

        # --- Build the table row by row ---

        for segment_label, segment_id in segments.items():
            md += f"| **{segment_label}** | |\\n" 

            current_dimensions = base_dimensions.copy()
            if segment_id != 'ifrs-full:OperatingSegmentsMember':
                 current_dimensions['ifrs-full:SegmentConsolidationItemsAxis'] = segment_id

            for item_id, item_label in line_items.items():
                value = self._get_fact_value(item_id, period_end_date, current_dimensions)
                amount_str = f"{value:,.0f}" if value is not None else ""
                md += f"| &nbsp;&nbsp;&nbsp;&nbsp;{item_label} | {amount_str} |\\n"
        
        return md.replace('\\n', '\n') 