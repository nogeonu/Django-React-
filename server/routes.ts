import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import { insertVisualSearchSchema, insertSearchHistorySchema } from "@shared/schema";
import { z } from "zod";

export async function registerRoutes(app: Express): Promise<Server> {
  // Get all products
  app.get("/api/products", async (req, res) => {
    try {
      const category = req.query.category as string;
      const products = category 
        ? await storage.getProductsByCategory(category)
        : await storage.getAllProducts();
      res.json(products);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch products" });
    }
  });

  // Get product by ID
  app.get("/api/products/:id", async (req, res) => {
    try {
      const product = await storage.getProductById(req.params.id);
      if (!product) {
        return res.status(404).json({ message: "Product not found" });
      }
      res.json(product);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch product" });
    }
  });

  // Perform visual search
  app.post("/api/visual-search", async (req, res) => {
    try {
      const { imageData } = insertVisualSearchSchema.parse(req.body);
      
      // Perform the visual search
      const results = await storage.performVisualSearch(imageData);
      
      // Save the search
      await storage.saveVisualSearch({
        imageData,
        results: JSON.stringify(results.map(r => ({ id: r.id, similarity: r.similarityScore })))
      });

      res.json(results);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ message: "Invalid request data", errors: error.errors });
      }
      res.status(500).json({ message: "Visual search failed" });
    }
  });

  // Get search history
  app.get("/api/search-history", async (req, res) => {
    try {
      const limit = req.query.limit ? parseInt(req.query.limit as string) : 10;
      const history = await storage.getSearchHistory(limit);
      res.json(history);
    } catch (error) {
      res.status(500).json({ message: "Failed to fetch search history" });
    }
  });

  // Add to search history
  app.post("/api/search-history", async (req, res) => {
    try {
      const data = insertSearchHistorySchema.parse(req.body);
      const history = await storage.addToSearchHistory(data);
      res.json(history);
    } catch (error) {
      if (error instanceof z.ZodError) {
        return res.status(400).json({ message: "Invalid request data", errors: error.errors });
      }
      res.status(500).json({ message: "Failed to add to search history" });
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}
