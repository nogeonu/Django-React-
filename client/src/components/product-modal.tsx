import { X, MapPin, Navigation, Bookmark } from "lucide-react";
import { type Product } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";

interface ProductModalProps {
  product: Product;
  onClose: () => void;
}

export default function ProductModal({ product, onClose }: ProductModalProps) {
  const { toast } = useToast();

  const formatPrice = (priceInCents: number) => {
    return `$${(priceInCents / 100).toFixed(0)}`;
  };

  const handleGetDirections = () => {
    toast({
      title: "Directions",
      description: `Navigate to ${product.location}`,
    });
  };

  const handleSaveProduct = () => {
    toast({
      title: "Saved",
      description: `${product.name} has been saved to your favorites`,
    });
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-end sm:items-center justify-center p-4"
      onClick={handleBackdropClick}
      data-testid="modal-backdrop"
    >
      <div 
        className="bg-white rounded-t-3xl sm:rounded-3xl w-full max-w-2xl max-h-[90vh] overflow-y-auto animate-scale-in"
        onClick={(e) => e.stopPropagation()}
        data-testid="modal-content"
      >
        <div className="p-6">
          {/* Modal Header */}
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-2xl font-bold text-slate-900" data-testid="text-modal-title">
              {product.name}
            </h3>
            <button 
              className="text-slate-400 hover:text-slate-600 p-2"
              onClick={onClose}
              data-testid="button-close-modal"
            >
              <X className="w-6 h-6" />
            </button>
          </div>
          
          {/* Product Image */}
          <div className="aspect-video bg-slate-100 rounded-xl mb-6 overflow-hidden">
            <img 
              src={product.imageUrl} 
              alt={product.name}
              className="w-full h-full object-cover"
              data-testid="img-modal-product"
            />
          </div>
          
          {/* Product Info */}
          <div className="space-y-4">
            <div>
              <h4 className="font-semibold text-slate-900 mb-2">Description</h4>
              <p className="text-slate-600" data-testid="text-modal-description">
                {product.fullDescription || product.description}
              </p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold text-slate-900 mb-1">Location</h4>
                <p className="text-slate-600 flex items-center">
                  <MapPin className="w-4 h-4 mr-2 text-primary" />
                  <span data-testid="text-modal-location">{product.location}</span>
                </p>
              </div>
              <div>
                <h4 className="font-semibold text-slate-900 mb-1">Price</h4>
                <p 
                  className="text-2xl font-bold text-primary"
                  data-testid="text-modal-price"
                >
                  {formatPrice(product.price)}
                </p>
              </div>
            </div>

            {/* Availability Status */}
            <div>
              <h4 className="font-semibold text-slate-900 mb-1">Availability</h4>
              <span 
                className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                  product.availability === "available" 
                    ? "bg-green-100 text-green-800"
                    : product.availability === "limited"
                    ? "bg-yellow-100 text-yellow-800"
                    : "bg-red-100 text-red-800"
                }`}
                data-testid="status-modal-availability"
              >
                {product.availability === "available" && "Available"}
                {product.availability === "limited" && "Limited Stock"}
                {product.availability === "sold_out" && "Sold Out"}
              </span>
            </div>
            
            {/* Action Buttons */}
            <div className="flex space-x-3 pt-4">
              <button 
                className="flex-1 bg-primary text-white py-3 px-6 rounded-xl font-semibold hover:bg-blue-600 transition-colors flex items-center justify-center space-x-2"
                onClick={handleGetDirections}
                data-testid="button-get-directions"
              >
                <Navigation className="w-5 h-5" />
                <span>Get Directions</span>
              </button>
              <button 
                className="flex-1 bg-slate-100 text-slate-700 py-3 px-6 rounded-xl font-semibold hover:bg-slate-200 transition-colors flex items-center justify-center space-x-2"
                onClick={handleSaveProduct}
                data-testid="button-save-product"
              >
                <Bookmark className="w-5 h-5" />
                <span>Save Item</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
