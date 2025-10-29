/**
 * Compress an image to reduce file size while maintaining quality
 * @param dataURL - The original image data URL
 * @param quality - Compression quality (0-1)
 * @param maxWidth - Maximum width for the compressed image
 * @returns Promise<string> - Compressed image data URL
 */
export const compressImage = (dataURL: string, quality: number = 0.7, maxWidth: number = 800): Promise<string> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Calculate new dimensions
      const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
      canvas.width = img.width * ratio;
      canvas.height = img.height * ratio;
      
      // Draw and compress
      ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
      const compressedDataURL = canvas.toDataURL('image/jpeg', quality);
      resolve(compressedDataURL);
    };
    
    img.src = dataURL;
  });
};

/**
 * Convert data URL to base64 string (without the data:image/jpeg;base64, prefix)
 * @param dataURL - The data URL to convert
 * @returns string - Base64 encoded string
 */
export const dataURLToBase64 = (dataURL: string): string => {
  return dataURL.split(',')[1] || '';
};

/**
 * Convert base64 string to data URL
 * @param base64 - Base64 encoded string
 * @param mimeType - MIME type (default: image/jpeg)
 * @returns string - Data URL
 */
export const base64ToDataURL = (base64: string, mimeType: string = 'image/jpeg'): string => {
  return `data:${mimeType};base64,${base64}`;
};

/**
 * Validate if the uploaded file is an image
 * @param file - File object to validate
 * @returns boolean - True if file is an image
 */
export const isImageFile = (file: File): boolean => {
  return file.type.startsWith('image/');
};

/**
 * Get image dimensions from a data URL
 * @param dataURL - The image data URL
 * @returns Promise<{width: number, height: number}> - Image dimensions
 */
export const getImageDimensions = (dataURL: string): Promise<{width: number, height: number}> => {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.width, height: img.height });
    };
    img.onerror = reject;
    img.src = dataURL;
  });
};

/**
 * Create a thumbnail from an image data URL
 * @param dataURL - Original image data URL
 * @param thumbnailSize - Size of the thumbnail (width and height)
 * @returns Promise<string> - Thumbnail data URL
 */
export const createThumbnail = (dataURL: string, thumbnailSize: number = 100): Promise<string> => {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      canvas.width = thumbnailSize;
      canvas.height = thumbnailSize;
      
      // Calculate crop dimensions to maintain aspect ratio
      const size = Math.min(img.width, img.height);
      const x = (img.width - size) / 2;
      const y = (img.height - size) / 2;
      
      ctx?.drawImage(img, x, y, size, size, 0, 0, thumbnailSize, thumbnailSize);
      const thumbnailDataURL = canvas.toDataURL('image/jpeg', 0.8);
      resolve(thumbnailDataURL);
    };
    
    img.src = dataURL;
  });
};
