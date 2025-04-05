
const cacheKeyPrefix: string = 'image-cache-';

// Cache an image src by its ID
export function cacheImage(id: string, src: string): void {
    // Generate the cache key using the image id
    const cacheKey = `${cacheKeyPrefix}${id}`;
    localStorage.setItem(cacheKey, src);
    console.log(`Image with ID ${id} cached.`);
  }

// Retrieve a cached image src by its ID
export function getCachedImage(id: string): string | null {
    const cacheKey = `${cacheKeyPrefix}${id}`;
    const image = localStorage.getItem(cacheKey);
    console.log(`Image with ID ${id} retrieved from cache`, image?.substring(0, 10));
    return image
}

export function clearImageCache(): void {
    // Clear all cached images
    Object.keys(localStorage).forEach((key) => {
        if (key.startsWith(cacheKeyPrefix)) {
            localStorage.removeItem(key);
        }
    });
    console.log('Cache cleared.');
}