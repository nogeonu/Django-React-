import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { MapPin } from "lucide-react";
import { type Product, type ProductWithSimilarity } from "@shared/schema";

interface ProductGridProps {
  onProductSelect: (product: Product) => void;
  searchResults: Product[];
  isSearching: boolean;
}

export default function ProductGrid({ onProductSelect, searchResults, isSearching }: ProductGridProps) {
  const [selectedCategory, setSelectedCategory] = useState("all");
  
  const { data: products = [], isLoading } = useQuery({
    queryKey: ["/api/products", selectedCategory],
    enabled: !isSearching && searchResults.length === 0,
  });

  const displayProducts = searchResults.length > 0 ? searchResults : products;

  const categories = [
    { id: "all", label: "All Items" },
    { id: "electronics", label: "Electronics" },
    { id: "fashion", label: "Fashion" },
    { id: "home-living", label: "Home & Living" },
    { id: "art-crafts", label: "Art & Crafts" },
  ];

  const getStatusBadge = (availability: string) => {
    switch (availability) {
      case "available":
        return <span className="status-available">Available</span>;
      case "limited":
        return <span className="status-limited">Limited</span>;
      case "sold_out":
        return <span className="status-sold-out">Sold Out</span>;
      default:
        return <span className="status-available">Available</span>;
    }
  };

  const getSimilarityBadge = (product: ProductWithSimilarity) => {
    if ('similarityScore' in product) {
      return (
        <span className="similarity-badge">
          {Math.round(product.similarityScore * 100)}% Match
        </span>
      );
    }
    return null;
  };

  const formatPrice = (priceInCents: number) => {
    return `$${(priceInCents / 100).toFixed(0)}`;
  };

  if (isLoading || isSearching) {
    return (
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, index) => (
            <div key={index} className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden animate-pulse">
              <div className="aspect-square bg-slate-200"></div>
              <div className="p-4 space-y-2">
                <div className="h-4 bg-slate-200 rounded"></div>
                <div className="h-3 bg-slate-200 rounded w-3/4"></div>
                <div className="flex justify-between items-center">
                  <div className="h-3 bg-slate-200 rounded w-1/2"></div>
                  <div className="h-4 bg-slate-200 rounded w-1/4"></div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </main>
    );
  }

  return (
    <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Search Results Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-slate-900 mb-2">
          {searchResults.length > 0 ? "Search Results" : "Discover Products"}
        </h2>
        <p className="text-slate-600">
          {searchResults.length > 0 
            ? `Found ${searchResults.length} matching products`
            : "Browse featured items available at the event"
          }
        </p>
      </div>

      {/* Filter Bar - Only show when not displaying search results */}
      {searchResults.length === 0 && (
        <div className="mb-6 flex flex-wrap gap-3">
          {categories.map((category) => (
            <button
              key={category.id}
              onClick={() => setSelectedCategory(category.id)}
              className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
                selectedCategory === category.id
                  ? "bg-primary text-white"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
              data-testid={`filter-${category.id}`}
            >
              {category.label}
            </button>
          ))}
        </div>
      )}

      {/* Product Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {displayProducts.map((product) => (
          <div
            key={product.id}
            className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden hover:shadow-lg transition-all transform hover:scale-105 cursor-pointer animate-fade-in"
            onClick={() => onProductSelect(product)}
            data-testid={`product-card-${product.id}`}
          >
            <div className="aspect-square bg-slate-100 relative overflow-hidden">
              <img
                src={product.imageUrl}
                alt={product.name}
                className="w-full h-full object-cover"
                loading="lazy"
              />
              <div className="absolute top-3 right-3">
                {getStatusBadge(product.availability)}
              </div>
              {getSimilarityBadge(product as ProductWithSimilarity) && (
                <div className="absolute top-3 left-3">
                  {getSimilarityBadge(product as ProductWithSimilarity)}
                </div>
              )}
            </div>
            <div className="p-4">
              <h3 className="font-semibold text-slate-900 mb-1 line-clamp-1" data-testid={`text-product-name-${product.id}`}>
                {product.name}
              </h3>
              <p className="text-sm text-slate-600 mb-2 line-clamp-2" data-testid={`text-product-description-${product.id}`}>
                {product.description}
              </p>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-500 flex items-center">
                  <MapPin className="w-3 h-3 mr-1" />
                  <span data-testid={`text-product-location-${product.id}`}>{product.location}</span>
                </span>
                <span 
                  className={`text-lg font-bold ${
                    product.availability === "sold_out" ? "text-slate-400" : "text-primary"
                  }`}
                  data-testid={`text-product-price-${product.id}`}
                >
                  {formatPrice(product.price)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {displayProducts.length === 0 && !isLoading && (
        <div className="text-center py-12">
          <p className="text-slate-500 text-lg">No products found</p>
          <p className="text-slate-400 text-sm mt-2">Try adjusting your search or filters</p>
        </div>
      )}

      {/* Load More Button - Only show for regular browsing */}
      {searchResults.length === 0 && displayProducts.length > 0 && (
        <div className="text-center mt-8">
          <button 
            className="bg-slate-100 text-slate-700 px-6 py-3 rounded-xl font-medium hover:bg-slate-200 transition-colors"
            data-testid="button-load-more"
          >
            Load More Products
          </button>
        </div>
      )}
    </main>
  );
}
