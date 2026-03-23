import pandas as pd

def generate_neighbor_list(gff_file, output_file):
    # Load GFF file, skipping comment lines
    df = pd.read_csv(gff_file, sep='\t', comment='#', header=None)
    df.columns = ['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
    
    # Filter for genes and parse locus_tag from attributes
    genes = df[df['type'] == 'gene'].copy()
    genes['locus_tag'] = genes['attributes'].str.extract('locus_tag=([^;]+)')
    
    # Sort by genomic position
    genes = genes.sort_values('start')
    
    neighbors = []
    
    # Iterate to find neighbors on the same strand with minimal distance
    for i in range(1, len(genes)):
        prev = genes.iloc[i-1]
        curr = genes.iloc[i]
        
        # Operon constraint: Same strand AND small intergenic region (e.g., < 200bp)
        if prev['strand'] == curr['strand']:
            distance = curr['start'] - prev['end']
            if 0 <= distance < 200:
                neighbors.append({'gene_1': prev['locus_tag'], 'gene_2': curr['locus_tag']})
    
    # Save to CSV for the model to load later
    neighbor_df = pd.DataFrame(neighbors)
    neighbor_df.to_csv(output_file, index=False)
    print(f"Successfully saved {len(neighbors)} neighbor pairs to {output_file}")

# Run the script
generate_neighbor_list('pountain_data/reference/GCF_000005845.2_ASM584v2_genomic.gff', 'operon_neighbors.csv')