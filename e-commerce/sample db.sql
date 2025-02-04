-- Create Users Table
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Addresses Table
CREATE TABLE addresses (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    address_line1 VARCHAR(255) NOT NULL,
    address_line2 VARCHAR(255),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(100) NOT NULL
);

-- Create Categories Table
CREATE TABLE categories (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- Create Products Table
CREATE TABLE products (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INT NOT NULL,
    category_id UUID REFERENCES categories(id),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Orders Table
CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    shipping_address_id UUID REFERENCES addresses(id),
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Order Items Table
CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    product_id UUID REFERENCES products(id),
    quantity INT NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL
);

-- Create Inventory Table
CREATE TABLE inventory (
    id UUID PRIMARY KEY,
    product_id UUID REFERENCES products(id),
    quantity INT NOT NULL,
    restocked_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Reviews Table
CREATE TABLE reviews (
    id UUID PRIMARY KEY,
    product_id UUID REFERENCES products(id),
    user_id UUID REFERENCES users(id),
    rating INT CHECK(rating >= 1 AND rating <= 5),
    comment TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Coupons Table
CREATE TABLE coupons (
    id UUID PRIMARY KEY,
    code VARCHAR(50) UNIQUE NOT NULL,
    discount DECIMAL(5, 2) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Create Shipments Table
CREATE TABLE shipments (
    id UUID PRIMARY KEY,
    order_id UUID REFERENCES orders(id),
    status VARCHAR(50) NOT NULL,
    carrier VARCHAR(100),
    tracking_number VARCHAR(100),
    shipped_at TIMESTAMP,
    delivered_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Insert Users (Sample Entries)
INSERT INTO users (id, name, email, password_hash, created_at, updated_at) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'John Doe', 'johndoe@example.com', 'hashedpassword123', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('550e8400-e29b-41d4-a716-446655440002', 'Jane Smith', 'janesmith@example.com', 'hashedpassword456', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('550e8400-e29b-41d4-a716-446655440003', 'Alice Johnson', 'alicejohnson@example.com', 'hashedpassword789', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('550e8400-e29b-41d4-a716-446655440004', 'Bob Lee', 'boblee@example.com', 'hashedpassword987', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('550e8400-e29b-41d4-a716-446655440005', 'Charlie Brown', 'charliebrown@example.com', 'hashedpassword123', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Addresses (Sample Entries)
INSERT INTO addresses (id, user_id, address_line1, city, state, postal_code, country) VALUES
    ('650e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', '123 Main St', 'New York', 'NY', '10001', 'USA'),
    ('650e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', '456 Elm St', 'Los Angeles', 'CA', '90001', 'USA'),
    ('650e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', '789 Oak St', 'Chicago', 'IL', '60001', 'USA'),
    ('650e8400-e29b-41d4-a716-446655440004', '550e8400-e29b-41d4-a716-446655440004', '321 Pine St', 'Dallas', 'TX', '75001', 'USA'),
    ('650e8400-e29b-41d4-a716-446655440005', '550e8400-e29b-41d4-a716-446655440005', '654 Birch St', 'San Francisco', 'CA', '94101', 'USA');

-- Insert Categories (Sample Entries)
INSERT INTO categories (id, name) VALUES
    ('750e8400-e29b-41d4-a716-446655440001', 'Electronics'),
    ('750e8400-e29b-41d4-a716-446655440002', 'Clothing'),
    ('750e8400-e29b-41d4-a716-446655440003', 'Home Appliances'),
    ('750e8400-e29b-41d4-a716-446655440004', 'Beauty & Personal Care'),
    ('750e8400-e29b-41d4-a716-446655440005', 'Sports & Outdoors');

-- Insert Products (Sample Entries)
INSERT INTO products (id, name, description, price, stock_quantity, category_id, created_at, updated_at) VALUES
    ('850e8400-e29b-41d4-a716-446655440001', 'Smartphone', 'Latest model smartphone with high-resolution camera', 699.99, 50, '750e8400-e29b-41d4-a716-446655440001', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('850e8400-e29b-41d4-a716-446655440002', 'Jeans', 'Comfortable denim jeans in various sizes', 39.99, 100, '750e8400-e29b-41d4-a716-446655440002', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('850e8400-e29b-41d4-a716-446655440003', 'Air Conditioner', 'Energy-efficient AC with remote control', 499.99, 30, '750e8400-e29b-41d4-a716-446655440003', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('850e8400-e29b-41d4-a716-446655440004', 'Skincare Cream', 'Moisturizing cream for dry skin', 19.99, 150, '750e8400-e29b-41d4-a716-446655440004', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('850e8400-e29b-41d4-a716-446655440005', 'Camping Tent', 'Waterproof tent for outdoor use', 89.99, 80, '750e8400-e29b-41d4-a716-446655440005', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Orders (Sample Entries)
INSERT INTO orders (id, user_id, status, total_amount, shipping_address_id, created_at, updated_at) VALUES
    ('950e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440001', 'Shipped', 799.98, '650e8400-e29b-41d4-a716-446655440001', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('950e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440002', 'Pending', 1199.97, '650e8400-e29b-41d4-a716-446655440002', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('950e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440003', 'Delivered', 2199.95, '650e8400-e29b-41d4-a716-446655440003', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Order Items (Sample Entries)
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
    ('a50e8400-e29b-41d4-a716-446655440001', '950e8400-e29b-41d4-a716-446655440001', '850e8400-e29b-41d4-a716-446655440001', 1, 699.99),
    ('a50e8400-e29b-41d4-a716-446655440002', '950e8400-e29b-41d4-a716-446655440002', '850e8400-e29b-41d4-a716-446655440002', 3, 39.99),
    ('a50e8400-e29b-41d4-a716-446655440003', '950e8400-e29b-41d4-a716-446655440003', '850e8400-e29b-41d4-a716-446655440003', 2, 499.99);

-- Insert Inventory (Sample Entries)
INSERT INTO inventory (id, product_id, quantity, restocked_at, created_at, updated_at) VALUES
    ('b50e8400-e29b-41d4-a716-446655440001', '850e8400-e29b-41d4-a716-446655440001', 50, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('b50e8400-e29b-41d4-a716-446655440002', '850e8400-e29b-41d4-a716-446655440002', 200, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('b50e8400-e29b-41d4-a716-446655440003', '850e8400-e29b-41d4-a716-446655440003', 100, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Reviews (Sample Entries)
INSERT INTO reviews (id, product_id, user_id, rating, comment, created_at, updated_at) VALUES
    ('c50e8400-e29b-41d4-a716-446655440001', '850e8400-e29b-41d4-a716-446655440001', '550e8400-e29b-41d4-a716-446655440002', 4, 'Good smartphone, fast performance', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('c50e8400-e29b-41d4-a716-446655440002', '850e8400-e29b-41d4-a716-446655440002', '550e8400-e29b-41d4-a716-446655440003', 5, 'Comfortable jeans! Great fit', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('c50e8400-e29b-41d4-a716-446655440003', '850e8400-e29b-41d4-a716-446655440003', '550e8400-e29b-41d4-a716-446655440004', 3, 'The AC works fine but noisy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Coupons (Sample Entries)
INSERT INTO coupons (id, code, discount, start_date, end_date, created_at, updated_at) VALUES
    ('d50e8400-e29b-41d4-a716-446655440001', 'SUMMER20', 20.00, '2025-06-01', '2025-08-31', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('d50e8400-e29b-41d4-a716-446655440002', 'FALL15', 15.00, '2025-09-01', '2025-11-30', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('d50e8400-e29b-41d4-a716-446655440003', 'WINTER25', 25.00, '2025-12-01', '2025-12-31', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

-- Insert Shipments (Sample Entries)
INSERT INTO shipments (id, order_id, status, carrier, tracking_number, shipped_at, delivered_at, created_at, updated_at) VALUES
    ('e50e8400-e29b-41d4-a716-446655440001', '950e8400-e29b-41d4-a716-446655440001', 'Shipped', 'UPS', '9876543210', CURRENT_TIMESTAMP, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('e50e8400-e29b-41d4-a716-446655440002', '950e8400-e29b-41d4-a716-446655440002', 'Delivered', 'FedEx', '1122334455', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
    ('e50e8400-e29b-41d4-a716-446655440003', '950e8400-e29b-41d4-a716-446655440003', 'Pending', 'DHL', '2233445566', CURRENT_TIMESTAMP, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
