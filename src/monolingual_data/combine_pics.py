from PIL import Image

def combine_images(image1_path, image2_path, output_path, gap=20):
    """
    Combine two PNG images horizontally with a gap and save the result.
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        output_path (str): Path where to save the combined image
        gap (int): Gap width in pixels between the two images
    """
    # Open images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    # Calculate dimensions for the combined image
    total_width = img1.width + img2.width + gap
    max_height = max(img1.height, img2.height)
    
    # Create new image with white background
    combined_img = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste images
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width + gap, 0))
    
    # Save result
    combined_img.save(output_path, 'PNG')
    print(f"Combined image saved to: {output_path}")

# Example usage with your plots
output_dir = '/netscratch/dgurgurov/projects2024/mt_lrls/corpus_analysis/plots/'

# Example usage with a gap of 50 pixels
combine_images(
    f'{output_dir}sentence_length_distributions.png',
    f'{output_dir}ttr_comparison.png',
    f'{output_dir}combined_analysis.png',
    gap=80
)

