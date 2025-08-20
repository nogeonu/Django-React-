# EventFind - Visual Product Search Application

## Overview

EventFind is a modern web application that enables users to discover products at events through visual search capabilities. Users can upload images to find similar products, browse product catalogs by category, and view detailed product information including location and availability at event booths.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React with TypeScript using Vite for development and bundling
- **UI Library**: Radix UI components with shadcn/ui for consistent design system
- **Styling**: Tailwind CSS with CSS variables for theming and responsive design
- **State Management**: TanStack Query (React Query) for server state management and caching
- **Routing**: Wouter for lightweight client-side routing
- **Form Handling**: React Hook Form with Zod validation for type-safe form management

### Backend Architecture
- **Runtime**: Node.js with Express.js server framework
- **Language**: TypeScript with ES modules for modern JavaScript features
- **API Design**: RESTful endpoints with JSON responses and proper HTTP status codes
- **Error Handling**: Centralized error middleware with structured error responses
- **Request Logging**: Custom middleware for API request tracking and performance monitoring

### Data Storage Solutions
- **Database**: PostgreSQL with Drizzle ORM for type-safe database operations
- **Connection**: Neon Database serverless PostgreSQL for cloud hosting
- **Schema Management**: Drizzle Kit for database migrations and schema evolution
- **Development Storage**: In-memory storage implementation for development and testing

### Authentication and Authorization
- **Session Management**: Express sessions with PostgreSQL session store (connect-pg-simple)
- **Security**: CORS configuration and request validation middleware
- **Development**: Currently uses in-memory authentication for development simplicity

### Key Features Implementation
- **Visual Search**: Image upload and processing for similarity-based product matching
- **Product Catalog**: Category-based product browsing with filtering capabilities
- **Search History**: Persistent storage of visual search queries and results
- **Product Details**: Modal-based product information display with booth location
- **Mobile Responsive**: Touch-friendly interface optimized for mobile event browsing

### Performance Optimizations
- **Image Processing**: Client-side image compression and base64 encoding
- **Caching**: React Query for intelligent data caching and background updates
- **Bundle Optimization**: Vite for fast development and optimized production builds
- **Code Splitting**: Dynamic imports for route-based code splitting

## External Dependencies

### Core Framework Dependencies
- **@tanstack/react-query**: Server state management and data fetching
- **wouter**: Lightweight routing library for single-page applications
- **react-hook-form** + **@hookform/resolvers**: Form handling with validation
- **zod**: Runtime type validation and schema definition

### UI and Styling Dependencies
- **@radix-ui/***: Comprehensive set of accessible UI primitives
- **tailwindcss**: Utility-first CSS framework for rapid styling
- **class-variance-authority**: Type-safe variant styling for components
- **lucide-react**: Modern icon library with React components

### Database and Backend Dependencies
- **drizzle-orm** + **drizzle-kit**: Type-safe ORM with automatic migrations
- **@neondatabase/serverless**: Serverless PostgreSQL client for Neon Database
- **connect-pg-simple**: PostgreSQL session store for Express sessions
- **express**: Web application framework for Node.js

### Development and Build Tools
- **vite**: Fast build tool and development server
- **typescript**: Static type checking for JavaScript
- **@replit/vite-plugin-runtime-error-modal**: Development error overlay
- **esbuild**: Fast JavaScript bundler for production builds

### Image Processing and Utilities
- **date-fns**: Modern date utility library for JavaScript
- **embla-carousel-react**: Touch-friendly carousel component
- **cmdk**: Command palette component for search interfaces