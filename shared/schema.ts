import { sql } from "drizzle-orm";
import { pgTable, text, varchar, integer, real, timestamp } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const products = pgTable("products", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  name: text("name").notNull(),
  description: text("description").notNull(),
  fullDescription: text("full_description"),
  price: integer("price").notNull(), // Price in cents
  location: text("location").notNull(),
  imageUrl: text("image_url").notNull(),
  category: text("category").notNull(),
  availability: text("availability").notNull().default("available"), // available, limited, sold_out
  createdAt: timestamp("created_at").defaultNow(),
});

export const searchHistory = pgTable("search_history", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  imageUrl: text("image_url").notNull(),
  searchResults: text("search_results").notNull(), // JSON string of product IDs
  timestamp: timestamp("timestamp").defaultNow(),
});

export const visualSearches = pgTable("visual_searches", {
  id: varchar("id").primaryKey().default(sql`gen_random_uuid()`),
  imageData: text("image_data").notNull(), // Base64 encoded image
  results: text("results").notNull(), // JSON string of matched products with similarity scores
  timestamp: timestamp("timestamp").defaultNow(),
});

export const insertProductSchema = createInsertSchema(products).omit({
  id: true,
  createdAt: true,
});

export const insertSearchHistorySchema = createInsertSchema(searchHistory).omit({
  id: true,
  timestamp: true,
});

export const insertVisualSearchSchema = createInsertSchema(visualSearches).omit({
  id: true,
  timestamp: true,
});

export type Product = typeof products.$inferSelect;
export type InsertProduct = z.infer<typeof insertProductSchema>;
export type SearchHistory = typeof searchHistory.$inferSelect;
export type InsertSearchHistory = z.infer<typeof insertSearchHistorySchema>;
export type VisualSearch = typeof visualSearches.$inferSelect;
export type InsertVisualSearch = z.infer<typeof insertVisualSearchSchema>;

// Extended types for frontend
export type ProductWithSimilarity = Product & {
  similarityScore: number;
};
