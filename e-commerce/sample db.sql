CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_category_id UUID REFERENCES categories(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INTEGER NOT NULL,
    category_id UUID NOT NULL REFERENCES categories(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE addresses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    address_line1 VARCHAR(255) NOT NULL,
    address_line2 VARCHAR(255),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    postal_code VARCHAR(20) NOT NULL,
    country VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    shipping_address_id UUID NOT NULL REFERENCES addresses(id),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES orders(id),
    product_id UUID NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert realistic sample data
-- Categories with realistic hierarchy
INSERT INTO categories (id, name, description, parent_category_id) VALUES
('11111111-1111-1111-1111-111111111111', 'Electronics', 'Latest electronic devices and accessories', NULL),
('22222222-2222-2222-2222-222222222222', 'Smartphones', 'Latest mobile phones and accessories', '11111111-1111-1111-1111-111111111111'),
('33333333-3333-3333-3333-333333333333', 'Laptops', 'Professional and gaming laptops', '11111111-1111-1111-1111-111111111111'),
('44444444-4444-4444-4444-444444444444', 'Computer Accessories', 'Peripherals and accessories', '11111111-1111-1111-1111-111111111111'),
('55555555-5555-5555-5555-555555555555', 'Home & Kitchen', 'Home appliances and kitchenware', NULL),
('66666666-6666-6666-6666-666666666666', 'Small Appliances', 'Compact kitchen appliances', '55555555-5555-5555-5555-555555555555'),
('77777777-7777-7777-7777-777777777777', 'Cookware', 'Pots, pans, and cooking accessories', '55555555-5555-5555-5555-555555555555');

-- Users with realistic names and emails
INSERT INTO users (id, name, email, password_hash) VALUES
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'Emily Johnson', 'emily.j@example.com', '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewKyBAWCHCL4Xlgu'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'Michael Chen', 'mchen@example.com', '$2a$12$glI.xJyC8bV3xz8U/XZkKOH.9xtiT2hH/ICuIHs2WgLxhx3ykQEKi'),
('cccccccc-cccc-cccc-cccc-cccccccccccc', 'Sarah Williams', 'swilliams@example.com', '$2a$12$QJigxHjCAl4ODxESOqbW7.swCQKadtnNkBg3rNGbgBYAZCqf7rt.m'),
('dddddddd-dddd-dddd-dddd-dddddddddddd', 'James Martinez', 'james.m@example.com', '$2a$12$BtlGxF9CAF8ENcAHqk0SD.Zl0K3RVp5w5bZGXvZ6O1OSNBFwgGOXO'),
('eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', 'Lisa Thompson', 'lisa.t@example.com', '$2a$12$kL9Ag3q4njI8NZxfEXHTF.XRF0bPj0vD5B4T5mHzLd.pi27brD6ji');

-- Products with realistic names, descriptions, and prices
INSERT INTO products (id, name, description, price, stock_quantity, category_id) VALUES
-- Smartphones
('ff111111-ffff-ffff-ffff-ffffffffffff', 'iPhone 15 Pro Max', '256GB, Space Black, A17 Pro chip', 1199.99, 85, '22222222-2222-2222-2222-222222222222'),
('ff222222-ffff-ffff-ffff-ffffffffffff', 'Samsung Galaxy S24 Ultra', '512GB, Titanium Gray, Snapdragon 8 Gen 3', 1299.99, 65, '22222222-2222-2222-2222-222222222222'),
('ff333333-ffff-ffff-ffff-ffffffffffff', 'Google Pixel 8 Pro', '256GB, Obsidian, Google Tensor G3', 999.99, 45, '22222222-2222-2222-2222-222222222222'),

-- Laptops
('ff444444-ffff-ffff-ffff-ffffffffffff', 'MacBook Pro 16"', 'M3 Max, 32GB RAM, 1TB SSD, Space Black', 2999.99, 30, '33333333-3333-3333-3333-333333333333'),
('ff555555-ffff-ffff-ffff-ffffffffffff', 'Dell XPS 15', 'Intel i9, 32GB RAM, 1TB SSD, RTX 4070', 2499.99, 25, '33333333-3333-3333-3333-333333333333'),
('ff666666-ffff-ffff-ffff-ffffffffffff', 'ASUS ROG Zephyrus', 'AMD Ryzen 9, 32GB RAM, 2TB SSD, RTX 4090', 3299.99, 20, '33333333-3333-3333-3333-333333333333'),

-- Computer Accessories
('ff777777-ffff-ffff-ffff-ffffffffffff', 'Logitech MX Master 3S', 'Advanced Wireless Mouse, Graphite', 99.99, 150, '44444444-4444-4444-4444-444444444444'),
('ff888888-ffff-ffff-ffff-ffffffffffff', 'Sony WH-1000XM5', 'Wireless Noise Cancelling Headphones, Black', 399.99, 100, '44444444-4444-4444-4444-444444444444'),

-- Kitchen Appliances
('ff999999-ffff-ffff-ffff-ffffffffffff', 'Ninja Foodi 9-in-1', 'Deluxe XL Pressure Cooker and Air Fryer', 249.99, 75, '66666666-6666-6666-6666-666666666666'),
('ffaaaaaa-ffff-ffff-ffff-ffffffffffff', 'KitchenAid Stand Mixer', 'Professional 5 Plus Series, 5 Quart', 449.99, 60, '66666666-6666-6666-6666-666666666666'),

-- Cookware
('ffbbbbbb-ffff-ffff-ffff-ffffffffffff', 'Le Creuset Dutch Oven', '5.5 Qt Round, Enameled Cast Iron, Flame', 369.99, 40, '77777777-7777-7777-7777-777777777777'),
('ffcccccc-ffff-ffff-ffff-ffffffffffff', 'All-Clad D5 Set', '10-Piece Stainless Steel Cookware Set', 899.99, 25, '77777777-7777-7777-7777-777777777777');

-- Addresses with realistic street names and postal codes
INSERT INTO addresses (id, user_id, address_line1, address_line2, city, state, postal_code, country) VALUES
('gg111111-gggg-gggg-gggg-gggggggggggg', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '2847 Madison Ave', 'Apt 4B', 'New York', 'NY', '10128', 'USA'),
('gg222222-gggg-gggg-gggg-gggggggggggg', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '1242 Beverly Blvd', NULL, 'Los Angeles', 'CA', '90048', 'USA'),
('gg333333-gggg-gggg-gggg-gggggggggggg', 'cccccccc-cccc-cccc-cccc-cccccccccccc', '742 N Michigan Ave', 'Unit 1601', 'Chicago', 'IL', '60611', 'USA'),
('gg444444-gggg-gggg-gggg-gggggggggggg', 'dddddddd-dddd-dddd-dddd-dddddddddddd', '3901 Lennox Ave', NULL, 'Seattle', 'WA', '98107', 'USA'),
('gg555555-gggg-gggg-gggg-gggggggggggg', 'eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', '1585 Boylston St', 'Apt 505', 'Boston', 'MA', '02115', 'USA');

-- Orders with realistic timestamps and statuses
INSERT INTO orders (id, user_id, status, total_amount, shipping_address_id, created_at) VALUES
('hh111111-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'DELIVERED', 3399.98, 'gg111111-gggg-gggg-gggg-gggggggggggg', '2024-01-10 14:23:54'),
('hh222222-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', 'PROCESSING', 2499.99, 'gg222222-gggg-gggg-gggg-gggggggggggg', '2024-01-12 09:45:12'),
('hh333333-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'cccccccc-cccc-cccc-cccc-cccccccccccc', 'SHIPPED', 1299.99, 'gg333333-gggg-gggg-gggg-gggggggggggg', '2024-01-11 16:30:00'),
('hh444444-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', 'DELIVERED', 449.99, 'gg111111-gggg-gggg-gggg-gggggggggggg', '2024-01-08 11:15:22'),
('hh555555-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'dddddddd-dddd-dddd-dddd-dddddddddddd', 'PENDING', 3699.98, 'gg444444-gggg-gggg-gggg-gggggggggggg', '2024-01-13 08:20:45');

-- Order items with realistic quantities
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price) VALUES
-- Order 1: iPhone + Headphones
('ii111111-iiii-iiii-iiii-iiiiiiiiiiii', 'hh111111-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ff111111-ffff-ffff-ffff-ffffffffffff', 1, 1199.99),
('ii222222-iiii-iiii-iiii-iiiiiiiiiiii', 'hh111111-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ff888888-ffff-ffff-ffff-ffffffffffff', 1, 399.99),

-- Order 2: Dell XPS 15
('ii333333-iiii-iiii-iiii-iiiiiiiiiiii', 'hh222222-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ff555555-ffff-ffff-ffff-ffffffffffff', 1, 2499.99),

-- Order 3: Galaxy S24 Ultra
('ii444444-iiii-iiii-iiii-iiiiiiiiiiii', 'hh333333-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ff222222-ffff-ffff-ffff-ffffffffffff', 1, 1299.99),

-- Order 4: KitchenAid Mixer
('ii555555-iiii-iiii-iiii-iiiiiiiiiiii', 'hh444444-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ffaaaaaa-ffff-ffff-ffff-ffffffffffff', 1, 449.99),

-- Order 5: MacBook Pro + Mouse
('ii666666-iiii-iiii-iiii-iiiiiiiiiiii', 'hh555555-hhhh-hhhh-hhhh-hhhhhhhhhhhh', 'ff444444-ffff-ffff-ffff-ffffffffffff', 1,449.99);