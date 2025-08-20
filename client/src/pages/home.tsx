import { useState } from "react";
import { Search, History, User } from "lucide-react";
import VisualSearch from "@/components/visual-search";
import ProductGrid from "@/components/product-grid";
import ProductModal from "@/components/product-modal";
import { type Product } from "@shared/schema";

export default function Home() {
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [searchResults, setSearchResults] = useState<Product[]>([]);
  const [isSearching, setIsSearching] = useState(false);

  const handleProductSelect = (product: Product) => {
    setSelectedProduct(product);
  };

  const handleCloseModal = () => {
    setSelectedProduct(null);
  };

  const handleSearchResults = (results: Product[]) => {
    setSearchResults(results);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Search className="text-primary text-2xl mr-3" />
              <h1 className="text-xl font-bold text-slate-900">EventFind</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button 
                className="p-2 text-slate-600 hover:text-primary transition-colors"
                data-testid="button-history"
              >
                <History className="w-5 h-5" />
              </button>
              <button 
                className="p-2 text-slate-600 hover:text-primary transition-colors"
                data-testid="button-profile"
              >
                <User className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Visual Search Section */}
      <VisualSearch 
        onSearchResults={handleSearchResults}
        isSearching={isSearching}
        setIsSearching={setIsSearching}
      />

      {/* Product Grid */}
      <ProductGrid 
        onProductSelect={handleProductSelect}
        searchResults={searchResults}
        isSearching={isSearching}
      />

      {/* Product Modal */}
      {selectedProduct && (
        <ProductModal 
          product={selectedProduct}
          onClose={handleCloseModal}
        />
      )}

      {/* Floating Action Button */}
      <button 
        className="fixed bottom-6 right-6 bg-primary text-white w-14 h-14 rounded-full shadow-lg hover:bg-blue-600 transition-all transform hover:scale-110 flex items-center justify-center z-40"
        data-testid="button-quick-camera"
      >
        <Search className="w-6 h-6" />
      </button>
    </div>
  );
}
