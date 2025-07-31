from UnifiedXBRLParser import UnifiedXBRLParser

import json
import os
import pandas as pd
import collections
from datetime import datetime
import re

class DocumentGenerator:
    """
    Generates structured, summarized, and grouped Markdown documents for each 
    financial concept based on the data from the UnifiedXBRLParser.
    This version incorporates advanced feedback for analytics and AI-readiness.
    """
    def __init__(self, unified_data):
        self.unified_data = unified_data
        self.file_paths = unified_data.get('__file_paths__', {})
        self._build_child_relationships()
        self._extract_segment_info()

    def _get_label_for_concept(self, concept_id, lang='en'):
        """Helper function to find the best available label for a concept."""
        if concept_id in self.unified_data:
            labels = self.unified_data[concept_id].get('labels', {})
            # Broader search for labels, including different roles
            label_priority = [
                f'taxonomy_{lang}_label', 
                f'instance_{lang}_label',
                f'{lang}_documentation',
                f'taxonomy_en_label', 
                f'instance_en_label',
                'en_documentation'
            ]
            for label_key in label_priority:
                if label_key in labels and labels[label_key]:
                    return labels[label_key]
        return concept_id

    def _build_child_relationships(self):
        """Pre-processes the data to map parent concepts to their children from all relationship types."""
        self.child_map = collections.defaultdict(list)
        for concept_id, data in self.unified_data.items():
            if not isinstance(data, dict): continue
            for rel_type, parents in data.get('relationships', {}).items():
                for parent_info in parents:
                    parent_id = parent_info.get('parent')
                    if parent_id:
                        self.child_map[parent_id].append(concept_id)

    def _extract_segment_info(self):
        """Extract segment members from the actual data rather than relying on parent-child relationships."""
        self.segment_members = {}
        
        # Look through all concepts and facts to find segment and geographic information
        for concept_id, data in self.unified_data.items():
            if not isinstance(data, dict): continue
            
            # Check if this is a custom revenue-related concept
            if isinstance(concept_id, str):
                if 'InternalRevenue' in concept_id:
                    # This is an internal revenue concept - note it for later use
                    data['is_internal_revenue'] = True
                elif 'NetRevenue' in concept_id:
                    # This is a net revenue concept - note it for later use
                    data['is_net_revenue'] = True
            
            facts = data.get('reported_facts', {}).get('numerical_facts', [])
            
            for fact in facts:
                dims = fact.get('context', {}).get('dimensions', {})
                segment_member = dims.get('Segment consolidation items [axis]')
                
                if segment_member and segment_member not in self.segment_members:
                    # Extract a cleaner label from the member ID
                    if 'BiopharmaceuticalMedicines' in segment_member:
                        self.segment_members[segment_member] = 'Biopharmaceutical Medicines'
                    elif 'ChemicalMedicines' in segment_member:
                        self.segment_members[segment_member] = 'Chemical Medicines'
                    elif 'All other segments' in segment_member:
                        self.segment_members[segment_member] = 'Other Segments'
                    elif 'Elimination' in segment_member or 'EliminationOfIntersegmentAmounts' in segment_member:
                        self.segment_members[segment_member] = 'Eliminations'
                    elif segment_member == 'Operating segments [member]':
                        self.segment_members[segment_member] = 'Operating Segments Total'
                    else:
                        # Use the member ID as-is if we can't identify it
                        self.segment_members[segment_member] = segment_member
                
                # Also store the fact's concept ID for easier lookup
                fact['concept_id'] = concept_id

    def _format_bignum(self, num):
        """Formats a large number into a human-readable string (e.g., 1.2T, 841.9B)."""
        if not isinstance(num, (int, float)): 
            return "N/A"
        if abs(num) >= 1e12: 
            return f"{num / 1e12:.1f}T KRW"
        if abs(num) >= 1e9: 
            return f"{num / 1e9:.1f}B KRW"
        if abs(num) >= 1e6: 
            return f"{num / 1e6:.1f}M KRW"
        return f"{num:,.0f} KRW"

    def _get_facts_by_context(self, facts, required_dims, required_decimals=None, exact_dims=False):
        """Utility to filter facts based on specific dimensions and decimals."""
        filtered_facts = {}
        for fact in facts:
            context = fact.get('context', {})
            dims = context.get('dimensions', {})
            
            # If exact_dims is True, the dimensions must be an exact match
            if exact_dims:
                if dims == required_dims:
                    match = True
                else:
                    match = False
            else: # Otherwise, use the more permissive check
                match = True
                for key, value in required_dims.items():
                    fact_value = dims.get(key)
                    if fact_value != value:
                        if not (value in str(fact_value) or str(fact_value) in value):
                            match = False
                            break
            
            if match:
                if required_decimals is None or fact.get('decimals') in required_decimals:
                    try:
                        period_str = context.get('period', '')
                        if ' to ' in period_str:
                            end_date_str = period_str.split(' to ')[-1]
                        else:
                            end_date_str = period_str
                        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
                        value = float(fact.get('value', 0))
                        
                        filtered_facts[end_date] = value
                    except (ValueError, IndexError):
                        continue
        return filtered_facts

    def _determine_period_type(self, date1, date2):
        """Determines the type of period comparison (YoY, QoQ, Annual, etc.)"""
        # Calculate the difference in days
        delta_days = abs((date1 - date2).days)
        
        # Check if same quarter different year (YoY quarterly)
        if 85 <= delta_days <= 95:  # Around 3 months
            if date1.month == date2.month:
                return "QoQ", "Quarter-over-Quarter"
            elif abs(date1.month - date2.month) in [0, 12]:
                return "YoY", "Year-over-Year"
        # Check if consecutive quarters
        elif 360 <= delta_days <= 370:  # Around 1 year
            return "YoY", "Year-over-Year"
        # Check if annual comparison
        elif 725 <= delta_days <= 740:  # Around 2 years
            return "YoY", "Year-over-Year (Annual)"
        # Semi-annual
        elif 175 <= delta_days <= 185:
            return "HoH", "Half-over-Half"
        else:
            # Generic period-over-period
            return "PoP", f"Period-over-Period ({delta_days} days)"
    
    def _format_period_label(self, date_obj, period_str=None):
        """Formats a period label based on the date and optional period string"""
        if period_str and ' to ' in period_str:
            # This is a period range
            start_str, end_str = period_str.split(' to ')
            try:
                start_date = datetime.strptime(start_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_str, '%Y-%m-%d')
                
                # Determine the period type
                days_diff = (end_date - start_date).days
                
                if 85 <= days_diff <= 95:  # Quarterly
                    quarter = (end_date.month - 1) // 3 + 1
                    return f"Q{quarter} {end_date.year}"
                elif 175 <= days_diff <= 185:  # Semi-annual
                    half = 1 if end_date.month <= 6 else 2
                    return f"H{half} {end_date.year}"
                elif 360 <= days_diff <= 370:  # Annual
                    return f"FY {end_date.year}"
                else:
                    # Custom period
                    return f"{start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}"
            except:
                pass
        
        # Default to quarter format if we just have a date
        quarter = (date_obj.month - 1) // 3 + 1
        return f"Q{quarter} {date_obj.year}"

    def _create_summary_tables(self, facts):
        """Creates comprehensive analytical summary tables."""
        if not self.segment_members:
            return "\n## Analytical Summary\n\nN/A: No segment data available for this concept.\n"

        # Don't filter by concept_id - use all facts passed to this method
        # The facts passed are already filtered to the specific concept
        
        base_dims = {'Consolidated and separate financial statements [axis]': 'Consolidated [member]'}
        
        # Try different decimal precisions, requiring an exact dimension match for the total
        total_facts = self._get_facts_by_context(facts, base_dims, required_decimals=['0'], exact_dims=True)
        if not total_facts:
            total_facts = self._get_facts_by_context(facts, base_dims, required_decimals=['-3'], exact_dims=True)
        
        if len(total_facts) < 2:
            # Fallback for concepts that only have segment data and no explicit total
            total_facts = self._get_facts_by_context(facts, base_dims, required_decimals=['-3'])
            if len(total_facts) < 2:
                return ""
        
        # Get the two most recent periods with their full period info
        period_info = {}
        for fact in facts:
            dims = fact.get('context', {}).get('dimensions', {})
            if dims == base_dims:
                try:
                    period_str = fact.get('context', {}).get('period', '')
                    end_date = datetime.strptime(period_str.split(' to ')[-1], '%Y-%m-%d')
                    period_info[end_date] = period_str
                except:
                    pass
        
        # Get the two most recent periods
        sorted_dates = sorted(total_facts.keys(), reverse=True)
        latest_period = sorted_dates[0]
        prev_period = sorted_dates[1]
        
        # Determine comparison type
        comp_type, comp_label = self._determine_period_type(latest_period, prev_period)
        
        # Get period labels
        latest_label = self._format_period_label(latest_period, period_info.get(latest_period))
        prev_label = self._format_period_label(prev_period, period_info.get(prev_period))
        
        # Calculate growth
        current_total = total_facts[latest_period]
        prev_total = total_facts[prev_period]
        total_growth = ((current_total - prev_total) / prev_total * 100) if prev_total else 0

        # Build the summary table
        table = "\n## Analytical Summary\n\n"
        
        # Add consolidated summary
        table += "**Consolidated Summary**\n"
        table += f"| Period | Total Value | {comp_label} Growth |\n"
        table += "|---|---|---|\n"
        table += f"| {latest_label} | {self._format_bignum(current_total)} | {total_growth:+.1f}% |\n"
        table += f"| {prev_label} | {self._format_bignum(prev_total)} | - |\n\n"
        
        # Get segment data
        segment_data = {}
        for member_id, label in self.segment_members.items():
            dims = {**base_dims, 'Segment consolidation items [axis]': member_id}
            segment_facts = self._get_facts_by_context(facts, dims, required_decimals=['-3'])
            if segment_facts:
                segment_data[label] = segment_facts

        if segment_data:
            # Find key insights
            insights = []
            for label, data in segment_data.items():
                if "Elimination" in label or "Total" in label or "Operating segments [member]" in label:
                    continue
                current = data.get(latest_period, 0)
                prev = data.get(prev_period, 0)
                if prev and current:
                    growth = ((current - prev) / prev * 100)
                    if abs(growth) > 10:  # Significant change
                        insights.append(f"- **{label}**: {growth:+.1f}% {comp_type} growth")

            if insights:
                table += "**Key Insights**\n"
                table += "\n".join(insights) + "\n\n"

            # Segment breakdown table
            table += "**Segment Performance Analysis**\n"
            table += f"| Segment | {latest_label} | % Contribution* | {prev_label} | {comp_label} Growth |\n"
            table += "|---|---|---|---|---|\n"
            
            # Calculate the sum of positive-value segments for a more intuitive contribution %
            sum_of_positive_segments = 0
            for label, data in segment_data.items():
                if label not in ['Operating Segments Total', 'Eliminations']:
                    value = data.get(latest_period, 0)
                    if value > 0:
                        sum_of_positive_segments += value

            # Define the order and filtering for segments
            segment_order = ['Biopharmaceutical Medicines', 'Chemical Medicines', 'Other Segments', 'Eliminations']
            
            for segment_name in segment_order:
                if segment_name in segment_data:
                    label = segment_name
                    data = segment_data[label]
                    current = data.get(latest_period, 0)
                    prev = data.get(prev_period, 0)
                    
                    if current == 0 and prev == 0:
                        continue  # Skip segments with no data
                    
                    contribution = (current / sum_of_positive_segments * 100) if sum_of_positive_segments else 0
                    growth = ((current - prev) / prev * 100) if prev else 0
                    
                    growth_str = f"{growth:+.1f}%"
                    if growth < 0:
                        growth_str = f"**{growth_str}**" # Bold for negative growth

                    if "Elimination" in label:
                        # Eliminations might be negative or positive depending on the concept
                        table += f"| *{label}* | *({self._format_bignum(abs(current))})* | N/A | *({self._format_bignum(abs(prev))})* | - |\n"
                    else:
                        table += f"| {label} | {self._format_bignum(current)} | {contribution:.1f}% | {self._format_bignum(prev)} | {growth_str} |\n"
            
            table += "\n*_% Contribution is calculated as a percentage of the sum of all positive-value segments before eliminations._\n"

        # For revenue concepts, add geographic analysis
        if any('revenue' in str(f.get('concept_id', '')).lower() for f in facts[:5]):  # Check first few facts
            geo_analysis = self._create_geographic_analysis(facts, latest_period, prev_period, comp_type, comp_label, latest_label, prev_label)
            if geo_analysis:
                table += "\n" + geo_analysis

        # --- Segment-based Calculation Validation ---
        validation_text = "\n**Segment-based Calculation Validation**\n"
        
        # Sum of individual segments, excluding any totals or eliminations
        sum_of_individual_segments = 0
        for label, data in segment_data.items():
            if label not in ['Operating Segments Total', 'Eliminations']:
                sum_of_individual_segments += data.get(latest_period, 0)

        # Get eliminations value separately
        eliminations_value = segment_data.get('Eliminations', {}).get(latest_period, 0)
        
        # The final calculated total is the sum of segments plus/minus eliminations
        calculated_total = sum_of_individual_segments + eliminations_value

        difference = current_total - calculated_total
        variance = (difference / current_total * 100) if current_total else 0
        
        validation_text += f"| Item | Value |\n"
        validation_text += f"|---|---|\n"
        validation_text += f"| Sum of Individual Segments | {self._format_bignum(sum_of_individual_segments)} |\n"
        validation_text += f"| Plus/Minus Eliminations | {self._format_bignum(eliminations_value)} |\n"
        validation_text += f"| **Calculated Total** | **{self._format_bignum(calculated_total)}** |\n"
        validation_text += f"| **Reported Total** | **{self._format_bignum(current_total)}** |\n"
        validation_text += f"| Difference (Variance) | {self._format_bignum(difference)} ({variance:.2f}%) |\n"
        validation_text += f"| Status | **{'Pass' if abs(variance) < 1 else 'Review Mismatch'}** |\n"
        
        table += validation_text
        
        return table

    def _create_geographic_analysis(self, facts, latest_period, prev_period, comp_type="YoY", comp_label="Year-over-Year", latest_label="Current", prev_label="Previous"):
        """Creates geographic revenue analysis using net (external) revenue where available."""
        base_dims = {'Consolidated and separate financial statements [axis]': 'Consolidated [member]'}
        
        geo_regions = {}
        for fact in facts:
            dims = fact.get('context', {}).get('dimensions', {})
            geo_member = dims.get('Geographical areas [axis]')
            if geo_member and geo_member not in geo_regions:
                # Create a readable label
                if 'Country of domicile' in geo_member:
                    geo_regions[geo_member] = 'Korea (Domestic)'
                elif 'Foreign countries' in geo_member:
                    geo_regions[geo_member] = 'Foreign Total'
                elif 'Asia' in geo_member and 'AsiaOf' in geo_member:
                    geo_regions[geo_member] = 'Asia'
                elif 'Europe' in geo_member:
                    geo_regions[geo_member] = 'Europe'
                elif 'NorthAmerica' in geo_member or 'North America' in geo_member:
                    geo_regions[geo_member] = 'North America'
                elif 'CentralAndSouthAmerica' in geo_member or 'Central' in geo_member:
                    geo_regions[geo_member] = 'Central/South America'
                else:
                    # Use a cleaned version of the member ID
                    label = geo_member.split(':')[-1].replace('OfGeographicalAreasMember', '')
                    geo_regions[geo_member] = label
        
        geo_data = {}
        for geo_member, label in geo_regions.items():
            current_val, prev_val = None, None
            
            # Prioritize facts that include the Operating Segments dimension, as this is more specific.
            dims_with_segment = {
                **base_dims, 
                'Geographical areas [axis]': geo_member,
                'Segment consolidation items [axis]': 'Operating segments [member]'
            }
            revenue_facts = self._get_facts_by_context(facts, dims_with_segment, required_decimals=['-3'], exact_dims=True)

            # If no segment-specific fact is found, fall back to the broader geographic fact.
            if not revenue_facts:
                dims_without_segment = {**base_dims, 'Geographical areas [axis]': geo_member}
                revenue_facts = self._get_facts_by_context(facts, dims_without_segment, required_decimals=['-3'], exact_dims=True)
            
            if revenue_facts:
                current_val = revenue_facts.get(latest_period)
                prev_val = revenue_facts.get(prev_period)

            if current_val is not None:
                geo_data[label] = {
                    'current': current_val,
                    'prev': prev_val if prev_val else 0
                }
        
        if not geo_data:
            return "\n**Geographic Revenue Distribution**\n\nN/A: No geographic breakdown available.\n"
        
        # Sort regions with Korea and Foreign Total first, then others
        def sort_key(item):
            label = item[0]
            if label == 'Korea (Domestic)':
                return '0'
            elif label == 'Foreign Total':
                return '1'
            else:
                return label
        
        sorted_geo_data = sorted(geo_data.items(), key=sort_key)
        
        table = "**Geographic Revenue Distribution**\n"
        table += f"| Region | {latest_label} | {prev_label} | {comp_label} Growth |\n"
        table += "|---|---|---|---|\n"
        
        for region, data in sorted_geo_data:
            # Skip the confusing "Foreign Total" as we are showing the breakdown
            if region == 'Foreign Total':
                continue
            if data['prev'] > 0:
                growth = ((data['current'] - data['prev']) / data['prev'] * 100)
                table += f"| {region} | {self._format_bignum(data['current'])} | {self._format_bignum(data['prev'])} | {growth:+.1f}% |\n"
            else:
                table += f"| {region} | {self._format_bignum(data['current'])} | - | N/A |\n"
        
        table += "\n*Note: Geographic revenues are based on gross sales from operating segments and may not sum to the net consolidated total due to inter-segment eliminations.*\n"
        
        return table

    def _group_and_format_facts(self, facts):
        """Groups facts by context and period, then formats them."""
        if not facts:
            return ""

        groups = collections.defaultdict(lambda: collections.defaultdict(set))
        for fact in facts:
            context = fact.get('context', {})
            dims = tuple(sorted(context.get('dimensions', {}).items()))
            period = context.get('period', 'N/A')
            fact_tuple = (fact.get('value'), fact.get('decimals'))
            groups[dims][period].add(fact_tuple)

        doc_section = "\n## Reported Numerical Facts\n"
        for dims_tuple, periods in sorted(groups.items()):
            
            # Create a title with both labels and raw IDs for better matching
            context_labels = " and ".join([f"{self._get_label_for_concept(dim)}: {self._get_label_for_concept(mem)}" for dim, mem in dims_tuple]) if dims_tuple else "Primary Context"
            context_ids = " and ".join([f"{dim}: {mem}" for dim, mem in dims_tuple]) if dims_tuple else ""
            
            doc_section += f"\n### **Context (Labels): {context_labels}**\n"
            if context_ids:
                doc_section += f"<!-- Context (IDs): {context_ids} -->\n" # Use a comment for raw IDs

            for period, fact_set in sorted(periods.items()):
                doc_section += f"- **Period: {period}**\n"
                for fact_tuple in fact_set:
                    value_str, decimals_str = fact_tuple
                    try:
                        value = float(value_str)
                        doc_section += f"  - **Value**: {value:,.0f} (decimals: {decimals_str})\n"
                    except (ValueError, TypeError):
                        doc_section += f"  - **Value**: {value_str}\n"
        return doc_section

    def _create_embeddable_text(self, doc_content):
        """Creates a concatenated string of all content for vectorization."""
        text = re.sub(r'#+\s', '', doc_content)
        text = re.sub(r'[\*\-`]', '', text)
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'\n{2,}', '\n', text)
        return "\n## Embeddable Text (for FAISS)\n```text\n" + text.strip() + "\n```"

    def create_concept_document(self, concept_id):
        """Creates a single, enhanced Markdown document for a given concept."""
        concept_data = self.unified_data.get(concept_id)
        if not concept_data: 
            return ""

        en_label = concept_data.get('labels', {}).get('taxonomy_en_label', concept_id)
        ko_label = concept_data.get('labels', {}).get('taxonomy_ko_label', '')
        
        doc = f"# Concept: {en_label}\n\n"
        doc += f"## Bilingual Labels\n"
        doc += f"- **EN**: {en_label}\n"
        if ko_label:
            doc += f"- **KO**: {ko_label}\n"
        
        ko_doc = concept_data.get('labels', {}).get('ko_documentation', '')
        en_doc = concept_data.get('labels', {}).get('en_documentation', '')

        if ko_doc:
            doc += f"- **KO Documentation**: {ko_doc}\n"
        if en_doc:
            doc += f"- **EN Documentation**: {en_doc}\n"
            
        doc += f"\n- **ID**: {concept_id}\n"
        
        if concept_id.startswith('entity00413046:'):
            base_concept = concept_id.split(':')[-1].split('Of')[0]
            doc += f"- **Note**: This is a company-specific extension, likely related to `ifrs-full:{base_concept}`.\n"

        # Get all facts for this concept
        numerical_facts = concept_data.get('reported_facts', {}).get('numerical_facts', [])
        
        # Check if this concept has segment data
        has_segment_data = any(
            'Segment consolidation items [axis]' in fact.get('context', {}).get('dimensions', {})
            for fact in numerical_facts
        )
        
        # Add analytical summary for concepts with segment data
        if has_segment_data and numerical_facts:
            summary = self._create_summary_tables(numerical_facts)
            if summary:  # Only add if summary was generated
                doc += summary

        # For revenue concepts, also look for related net revenue concepts
        if 'Revenue' in concept_id and concept_id == 'ifrs-full:Revenue':
            # Find all related revenue concepts (internal, net, etc.)
            for other_concept_id, other_data in self.unified_data.items():
                if not isinstance(other_data, dict): 
                    continue
                if ('Revenue' in str(other_concept_id) and 
                    other_concept_id != concept_id):
                    # Add these facts to our analysis
                    other_facts = other_data.get('reported_facts', {}).get('numerical_facts', [])
                    for fact in other_facts:
                        fact['concept_id'] = other_concept_id
                        fact['concept_type'] = 'NetRevenue' if 'Net' in str(other_concept_id) else 'InternalRevenue' if 'Internal' in str(other_concept_id) else 'Other'
                    numerical_facts.extend(other_facts)

        # Group only the facts for the main concept (not the extended ones)
        main_facts = [f for f in numerical_facts if f.get('concept_id', concept_id) == concept_id]
        doc += self._group_and_format_facts(main_facts)
        
        text_blocks = concept_data.get('reported_facts', {}).get('text_blocks', [])
        if text_blocks:
            doc += "\n## Explanatory Notes\n"
            for block in text_blocks:
                doc += f"```text\n{block.get('text', '')}\n```\n"
        
        references = concept_data.get('references', {}).get('accounting_standard', [])
        if references:
            doc += "\n## Accounting Standard References\n"
            for ref in references:
                doc += f"- **Standard**: {ref.get('name')} {ref.get('number')}, Paragraph: {ref.get('paragraph')}\n"
        
        if self.file_paths.get('instance'):
            doc += f"\n---\n*Source Document: {os.path.basename(self.file_paths['instance'])}*"

        doc += self._create_embeddable_text(doc)
        
        return doc

    def generate_all_documents(self):
        """Generates Markdown documents for all concepts."""
        all_docs = {}
        
        for concept_id in self.unified_data.keys():
            if not isinstance(self.unified_data[concept_id], dict): 
                continue
            
            concept_data = self.unified_data[concept_id]
            if concept_data.get('reported_facts'):
                try:
                    doc = self.create_concept_document(concept_id)
                    if doc:  # Only add if document was created
                        all_docs[concept_id] = doc
                except Exception as e:
                    print(f"ERROR: Failed to create document for {concept_id}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        print(f"Successfully generated {len(all_docs)} concept documents.")
        
        return all_docs


# --- Example Usage ---
if __name__ == '__main__':
    data_folder = './data' 
    file_paths = {
        'concepts': os.path.join(data_folder, 'Concepts.csv'),
        'labels_ko': os.path.join(data_folder, 'entity00413046_2025-03-31_lab-ko.xml'),
        'labels_en': os.path.join(data_folder, 'entity00413046_2025-03-31_lab-en.xml'),
        'presentation': os.path.join(data_folder, 'Presentation Link.csv'),
        'calculation': os.path.join(data_folder, 'Calculation Link.csv'),
        'instance': os.path.join(data_folder, 'entity00413046_2025-03-31.xbrl'),
        'taxonomy_labels': os.path.join(data_folder, 'Label Link.csv'),
        'references': os.path.join(data_folder, 'Reference Link.csv')
    }

    print("--- Running XBRL Parser ---")
    parser = UnifiedXBRLParser(file_paths)
    unified_data = parser.run_parser()
    print("-" * 100)

    print("\n--- Generating Enhanced Concept Documents ---")
    doc_generator = DocumentGenerator(unified_data)
    all_documents = doc_generator.generate_all_documents()
    print("-" * 100)

    print("\n--- Sample Enhanced Documents ---")

    print("\n--- Document for 'ifrs-full:Revenue' ---")
    print(all_documents.get('ifrs-full:Revenue', "Document not found."))
    print("-" * 100)
    
    print("\n--- Document for 'ifrs-full:DepreciationExpense' ---")
    print(all_documents.get('ifrs-full:DepreciationExpense', "Document not found."))
    print("-" * 100)
    
    # This custom concept is a good example of segment data
    print("\n--- Document for 'entity00413046:AmortisationExpenseOfDisclosureOfOperatingSegmentsLineItemsOfDisclosureOfOperatingSegmentsTableOfItems' ---")
    print(all_documents.get('entity00413046:AmortisationExpenseOfDisclosureOfOperatingSegmentsLineItemsOfDisclosureOfOperatingSegmentsTableOfItems', "Document not found."))