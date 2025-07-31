# main.py

import os
import pickle
from UnifiedXBRLParser import UnifiedXBRLParser
from DocumentGenerator import DocumentGenerator
from StatementGenerator import StatementGenerator

def main():
    """
    Main execution script to run the XBRL parser and document generator.
    """
    # --- Configuration ---
    # Define the folder where your data files are located.
    data_folder = './data'
    output_folder = './output/concept_details'
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the paths to all the necessary XBRL and taxonomy files.
    file_paths = {
        'concepts': os.path.join(data_folder, 'Concepts.csv'),
        'labels_ko': os.path.join(data_folder, 'entity00413046_2025-03-31_lab-ko.xml'),
        'labels_en': os.path.join(data_folder, 'entity00413046_2025-03-31_lab-en.xml'),
        'presentation_xml': os.path.join(data_folder, 'entity00413046_2024-12-31_pre.xml'),
        'calculation': os.path.join(data_folder, 'Calculation Link.csv'),
        'instance': os.path.join(data_folder, 'entity00413046_2025-03-31.xbrl'),
        'taxonomy_labels': os.path.join(data_folder, 'Label Link.csv'),
        'references': os.path.join(data_folder, 'Reference Link.csv'),
        'role_types': os.path.join(data_folder, 'RoleTypes.csv')
    }

    # --- Step 1: Run the Unified XBRL Parser ---
    print("--- Running Unified XBRL Parser ---")
    parser = UnifiedXBRLParser(file_paths)
    unified_data = parser.run_parser()
    
    # Add file paths to the data for reference in the generator
    unified_data['__file_paths__'] = file_paths
    print("-" * 50)

    # --- Step 2: Generate Enhanced Analytical Documents FIRST ---
    # This is now a prerequisite for the statement generator
    print("\n--- Generating Enhanced Analytical Documents ---")
    doc_generator = DocumentGenerator(unified_data)
    all_documents = doc_generator.generate_all_documents()
    print("-" * 50)

    # --- Step 3: Generate Core Financial Statements ---
    # This now uses the documents generated in the previous step
    print("\n--- Generating Core Financial Statements ---")
    statement_gen = StatementGenerator(parser, all_documents)
    period_end_date = '2025-03-31'
    
    # Generate the standard Income Statement
    income_statement_uri = 'http://dart.fss.or.kr/role/ifrs/dart_2024-06-30_role-D310000'
    income_dimensions = {'ifrs-full:ConsolidatedAndSeparateFinancialStatementsAxis': 'ifrs-full:ConsolidatedMember'}
    income_statement_md = statement_gen.generate_statement(income_statement_uri, period_end_date, income_dimensions)
    income_filename = "Consolidated_Income_Statement.md"
    income_filepath = os.path.join(output_folder, income_filename) # Save directly in output/
    with open(income_filepath, "w", encoding="utf-8") as f:
        f.write(income_statement_md)
    print(f"  - Generated: {income_filename}")

    # Generate the specialized Operating Segments report using the custom function
    op_segment_uri = 'http://dart.fss.or.kr/role/ifrs/ifrs_8_role-D871100'
    op_segment_md = statement_gen.generate_custom_operating_segments(op_segment_uri, period_end_date)
    op_segment_filename = "Disclosure_of_Operating_Segments.md"
    op_segment_filepath = os.path.join(output_folder, op_segment_filename) # Save directly in output/
    if op_segment_md:
        with open(op_segment_filepath, "w", encoding="utf-8") as f:
            f.write(op_segment_md)
        print(f"  - Generated: {op_segment_filename}")

    # Save Concept Documents into output/concept_details/
    concept_output_folder = os.path.join(output_folder, 'concept_details')
    os.makedirs(concept_output_folder, exist_ok=True)
    
    print(f"\n--- Saving {len(all_documents)} Documents to '{concept_output_folder}' ---")

    for concept_id, doc_content in all_documents.items():
        # Sanitize filename
        filename = concept_id.replace(":", "_").replace("/", "_")
        # Truncate long filenames
        if len(filename) > 200:
            filename = filename[:200]
        filename += ".md"
        
        filepath = os.path.join(concept_output_folder, filename) # Use the correct folder
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc_content)
    
    # Save all documents to a single pickle file for efficient loading elsewhere
    pickle_path = 'all_documents.pkl'
    print(f"\n--- Saving all documents to '{pickle_path}' ---")
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_documents, f)

    print("\n--- Process Finished Successfully ---")
    print(f"✅  {len(all_documents)} analytical documents have been generated and saved.")

if __name__ == '__main__':
    main()