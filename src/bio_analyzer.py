"""
Biopython Integration for Biological Data Analysis
Demonstrates working with biological sequences and data
"""

from Bio.Seq import Seq
from Bio.SeqUtils import molecular_weight, gc_fraction
from Bio import SeqIO
from io import StringIO
import random


class BiologicalDataAnalyzer:
    """Analyze biological sequences using Biopython"""
    
    def __init__(self):
        self.sequences = []
        
    def generate_sample_sequences(self, n_sequences: int = 10) -> list:
        """
        Generate sample DNA sequences for demonstration
        
        Args:
            n_sequences: Number of sequences to generate
            
        Returns:
            List of Seq objects
        """
        nucleotides = ['A', 'T', 'G', 'C']
        sequences = []
        
        for i in range(n_sequences):
            length = random.randint(50, 200)
            seq_str = ''.join(random.choices(nucleotides, k=length))
            seq = Seq(seq_str)
            sequences.append(seq)
        
        self.sequences = sequences
        return sequences
    
    def analyze_sequence(self, sequence: Seq) -> dict:
        """
        Analyze a biological sequence
        
        Args:
            sequence: Biopython Seq object
            
        Returns:
            Dictionary with sequence analysis
        """
        analysis = {
            'length': len(sequence),
            'gc_content': gc_fraction(sequence) * 100,
            'nucleotide_counts': {
                'A': sequence.count('A'),
                'T': sequence.count('T'),
                'G': sequence.count('G'),
                'C': sequence.count('C')
            }
        }
        
        # Get complement and reverse complement
        analysis['complement'] = str(sequence.complement())[:50] + '...'
        analysis['reverse_complement'] = str(sequence.reverse_complement())[:50] + '...'
        
        # Transcribe to RNA
        rna = sequence.transcribe()
        analysis['rna_sequence'] = str(rna)[:50] + '...'
        
        # Translate to protein (if possible)
        try:
            protein = sequence.translate()
            analysis['protein_sequence'] = str(protein)[:30] + '...'
        except:
            analysis['protein_sequence'] = 'N/A'
        
        return analysis
    
    def batch_analyze_sequences(self, sequences: list = None) -> dict:
        """
        Analyze multiple sequences
        
        Args:
            sequences: List of Seq objects (uses stored sequences if None)
            
        Returns:
            Dictionary with batch analysis results
        """
        if sequences is None:
            sequences = self.sequences
        
        if not sequences:
            raise ValueError("No sequences to analyze")
        
        gc_contents = [gc_fraction(seq) * 100 for seq in sequences]
        lengths = [len(seq) for seq in sequences]
        
        results = {
            'total_sequences': len(sequences),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'avg_gc_content': sum(gc_contents) / len(gc_contents),
            'min_gc_content': min(gc_contents),
            'max_gc_content': max(gc_contents),
            'sequences_analyzed': len(sequences)
        }
        
        return results
    
    def find_motif(self, sequence: Seq, motif: str) -> list:
        """
        Find a motif in a sequence
        
        Args:
            sequence: DNA sequence
            motif: Motif to search for
            
        Returns:
            List of positions where motif is found
        """
        seq_str = str(sequence)
        positions = []
        start = 0
        
        while True:
            pos = seq_str.find(motif, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        
        return positions
    
    def medical_genomics_summary(self) -> dict:
        """
        Generate a summary for medical genomics context
        
        Returns:
            Dictionary with genomics insights
        """
        if not self.sequences:
            self.generate_sample_sequences()
        
        batch_results = self.batch_analyze_sequences()
        
        # Medical context
        summary = {
            'genomic_analysis': batch_results,
            'interpretation': {
                'gc_content_significance': 'GC content can indicate genomic regions and gene density',
                'sequence_diversity': f"Analyzed {batch_results['total_sequences']} sequences with varying lengths",
                'medical_relevance': 'Sequence analysis is crucial for identifying genetic variants associated with diseases'
            }
        }
        
        return summary
