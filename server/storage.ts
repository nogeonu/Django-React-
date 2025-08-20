import { type Product, type InsertProduct, type SearchHistory, type InsertSearchHistory, type VisualSearch, type InsertVisualSearch, type ProductWithSimilarity } from "@shared/schema";
import { randomUUID } from "crypto";

export interface IStorage {
  // Products
  getAllProducts(): Promise<Product[]>;
  getProductById(id: string): Promise<Product | undefined>;
  getProductsByCategory(category: string): Promise<Product[]>;
  createProduct(product: InsertProduct): Promise<Product>;
  
  // Visual Search
  performVisualSearch(imageData: string): Promise<ProductWithSimilarity[]>;
  saveVisualSearch(search: InsertVisualSearch): Promise<VisualSearch>;
  
  // Search History
  getSearchHistory(limit?: number): Promise<SearchHistory[]>;
  addToSearchHistory(history: InsertSearchHistory): Promise<SearchHistory>;
}

export class MemStorage implements IStorage {
  private products: Map<string, Product>;
  private searchHistory: Map<string, SearchHistory>;
  private visualSearches: Map<string, VisualSearch>;

  constructor() {
    this.products = new Map();
    this.searchHistory = new Map();
    this.visualSearches = new Map();
    this.initializeSampleData();
  }

  private initializeSampleData() {
    const sampleProducts: InsertProduct[] = [
      {
        name: "Wireless Headphones Pro",
        description: "Premium noise-canceling headphones with superior sound quality",
        fullDescription: "Premium noise-canceling headphones with superior sound quality, 30-hour battery life, and comfortable over-ear design perfect for long listening sessions.",
        price: 29900, // $299
        location: "Booth A-12",
        imageUrl: "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "electronics",
        availability: "available"
      },
      {
        name: "Smart Fitness Watch",
        description: "Advanced fitness tracking with heart rate monitoring",
        fullDescription: "Advanced fitness tracking with heart rate monitoring, GPS, waterproof design, and week-long battery life for active lifestyles.",
        price: 19900, // $199
        location: "Booth C-5",
        imageUrl: "https://images.unsplash.com/photo-1523275335684-37898b6baf30?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "electronics",
        availability: "limited"
      },
      {
        name: "Handcrafted Ceramic Vase",
        description: "Unique artisan-made ceramic piece with organic patterns",
        fullDescription: "Unique artisan-made ceramic piece with organic patterns, perfect for home decoration and showcasing fresh flowers.",
        price: 8500, // $85
        location: "Booth B-8",
        imageUrl: "https://images.unsplash.com/photo-1578662996442-48f60103fc96?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "home-living",
        availability: "available"
      },
      {
        name: "Vintage Leather Backpack",
        description: "Premium leather backpack with classic design and durability",
        fullDescription: "Premium leather backpack with classic design and durability, featuring multiple compartments and comfortable straps.",
        price: 15900, // $159
        location: "Booth D-15",
        imageUrl: "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "fashion",
        availability: "available"
      },
      {
        name: "Modern Table Lamp",
        description: "Minimalist design with adjustable brightness and USB charging",
        fullDescription: "Minimalist design with adjustable brightness and USB charging port, perfect for modern workspaces and reading.",
        price: 12900, // $129
        location: "Booth A-7",
        imageUrl: "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "home-living",
        availability: "sold_out"
      },
      {
        name: "Abstract Art Print",
        description: "Limited edition print on premium paper with vibrant colors",
        fullDescription: "Limited edition print on premium paper with vibrant colors, professionally framed and ready to hang.",
        price: 7500, // $75
        location: "Booth E-3",
        imageUrl: "https://images.unsplash.com/photo-1541961017774-22349e4a1262?ixlib=rb-4.0.3&auto=format&fit=crop&w=400&h=400",
        category: "art-crafts",
        availability: "available"
      }
    ];

    sampleProducts.forEach(async (product) => {
      await this.createProduct(product);
    });
  }

  async getAllProducts(): Promise<Product[]> {
    return Array.from(this.products.values());
  }

  async getProductById(id: string): Promise<Product | undefined> {
    return this.products.get(id);
  }

  async getProductsByCategory(category: string): Promise<Product[]> {
    return Array.from(this.products.values()).filter(
      (product) => category === "all" || product.category === category
    );
  }

  async createProduct(insertProduct: InsertProduct): Promise<Product> {
    const id = randomUUID();
    const product: Product = {
      ...insertProduct,
      id,
      createdAt: new Date(),
    };
    this.products.set(id, product);
    return product;
  }

  async performVisualSearch(imageData: string): Promise<ProductWithSimilarity[]> {
    // Simulate image analysis and matching
    const allProducts = Array.from(this.products.values());
    
    // Simple similarity simulation based on product characteristics
    const results: ProductWithSimilarity[] = allProducts.map((product) => {
      // Generate similarity score based on various factors
      const baseScore = Math.random() * 0.3 + 0.7; // 70-100%
      const categoryBonus = product.category === "electronics" ? 0.05 : 0;
      const availabilityPenalty = product.availability === "sold_out" ? -0.1 : 0;
      
      const similarityScore = Math.min(0.99, Math.max(0.5, 
        baseScore + categoryBonus + availabilityPenalty
      ));

      return {
        ...product,
        similarityScore: Math.round(similarityScore * 100) / 100
      };
    });

    // Sort by similarity score descending
    return results.sort((a, b) => b.similarityScore - a.similarityScore);
  }

  async saveVisualSearch(search: InsertVisualSearch): Promise<VisualSearch> {
    const id = randomUUID();
    const visualSearch: VisualSearch = {
      ...search,
      id,
      timestamp: new Date(),
    };
    this.visualSearches.set(id, visualSearch);
    return visualSearch;
  }

  async getSearchHistory(limit: number = 10): Promise<SearchHistory[]> {
    const history = Array.from(this.searchHistory.values())
      .sort((a, b) => (b.timestamp?.getTime() || 0) - (a.timestamp?.getTime() || 0))
      .slice(0, limit);
    return history;
  }

  async addToSearchHistory(insertHistory: InsertSearchHistory): Promise<SearchHistory> {
    const id = randomUUID();
    const history: SearchHistory = {
      ...insertHistory,
      id,
      timestamp: new Date(),
    };
    this.searchHistory.set(id, history);
    return history;
  }
}

export const storage = new MemStorage();
