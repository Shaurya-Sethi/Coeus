-- Create the tables
CREATE TABLE users (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE orders (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    status VARCHAR(50) NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    shipping_address_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE products (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    price DECIMAL(10, 2) NOT NULL,
    stock_quantity INTEGER NOT NULL,
    category_id UUID NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE order_items (
    id UUID PRIMARY KEY,
    order_id UUID NOT NULL,
    product_id UUID NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE categories (
    id UUID PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    parent_category_id UUID
);

CREATE TABLE payments (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    order_id UUID NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    payment_method VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (order_id) REFERENCES orders(id)
);

CREATE TABLE reviews (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    product_id UUID NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (product_id) REFERENCES products(id)
);

CREATE TABLE inventory (
    id UUID PRIMARY KEY,
    product_id UUID NOT NULL,
    quantity INTEGER NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(id)
);

-- Insert data into users
INSERT INTO users (id, name, email, password_hash, created_at, updated_at)
VALUES
    ('b1a3c5d7-8e6f-4d76-a85a-e84db62a9e7a', 'Alice Johnson', 'alice.johnson@email.com', 'hashed_password_1', '2025-01-01 10:00:00', '2025-01-01 10:00:00'),
    ('f3d76c1a-1b63-4a6b-9a0e-fc51720e1b12', 'Bob Smith', 'bob.smith@email.com', 'hashed_password_2', '2025-01-02 11:00:00', '2025-01-02 11:00:00');

-- Insert data into categories
INSERT INTO categories (id, name, description, parent_category_id)
VALUES
    ('baae6c2e-22ed-4975-a09c-cd7d56c1227e', 'Electronics', 'Devices like phones, laptops, etc.', NULL),
    ('da28c876-9b5d-4d4c-9983-535a53024598', 'Home Appliances', 'Appliances for home use', NULL);

-- Insert data into products
INSERT INTO products (id, name, description, price, stock_quantity, category_id, created_at, updated_at)
VALUES
    ('bc7e29be-8fd2-4e7f-87a4-4c7c3a90a53f', 'Smartphone', 'Latest model smartphone', 799.99, 100, 'baae6c2e-22ed-4975-a09c-cd7d56c1227e', '2025-01-01 12:00:00', '2025-01-01 12:00:00'),
    ('7b9e8c92-0971-47c4-bdd3-e5162c8a7a39', 'Washing Machine', 'High efficiency washing machine', 599.99, 50, 'da28c876-9b5d-4d4c-9983-535a53024598', '2025-01-03 09:00:00', '2025-01-03 09:00:00');

-- Insert data into orders
INSERT INTO orders (id, user_id, status, total_amount, shipping_address_id, created_at, updated_at)
VALUES
    ('a145f981-b7e9-4e2d-9a2d-16550f689928', 'b1a3c5d7-8e6f-4d76-a85a-e84db62a9e7a', 'pending', 799.99, 'e7d2694d-8522-4027-9fe5-fbc18c8c3a2b', '2025-01-05 14:00:00', '2025-01-05 14:00:00'),
    ('e59736f2-2423-4ecf-87b0-8b1f5d1e6f10', 'f3d76c1a-1b63-4a6b-9a0e-fc51720e1b12', 'shipped', 599.99, '5c218469-4ac0-47e9-8c3f-d7e1cfdb5f8b', '2025-01-06 15:00:00', '2025-01-06 15:00:00');

-- Insert data into order_items
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price)
VALUES
    ('1eae9983-f1b5-4bfa-a018-4cd0e8001b89', 'a145f981-b7e9-4e2d-9a2d-16550f689928', 'bc7e29be-8fd2-4e7f-87a4-4c7c3a90a53f', 1, 799.99),
    ('3b7591ad-9b70-438b-b5ba-1070a88a3be6', 'e59736f2-2423-4ecf-87b0-8b1f5d1e6f10', '7b9e8c92-0971-47c4-bdd3-e5162c8a7a39', 1, 599.99);

-- Insert data into payments
INSERT INTO payments (id, user_id, order_id, amount, payment_method, status, created_at)
VALUES
    ('d743fe53-c1ea-4067-b40d-906c1fa25462', 'b1a3c5d7-8e6f-4d76-a85a-e84db62a9e7a', 'a145f981-b7e9-4e2d-9a2d-16550f689928', 799.99, 'credit card', 'completed', '2025-01-05 14:15:00'),
    ('d21b3b29-c739-4bfe-b071-34242fe1f324', 'f3d76c1a-1b63-4a6b-9a0e-fc51720e1b12', 'e59736f2-2423-4ecf-87b0-8b1f5d1e6f10', 599.99, 'PayPal', 'pending', '2025-01-06 15:15:00');

-- Insert data into reviews
INSERT INTO reviews (id, user_id, product_id, rating, comment)
VALUES
    ('f938e70a-dfa0-4558-a9b0-88ed94336c92', 'b1a3c5d7-8e6f-4d76-a85a-e84db62a9e7a', 'bc7e29be-8fd2-4e7f-87a4-4c7c3a90a53f', 5, 'Great smartphone, very fast and sleek!'),
    ('a4a56e42-799e-4599-bd13-949f9446db51', 'f3d76c1a-1b63-4a6b-9a0e-fc51720e1b12', '7b9e8c92-0971-47c4-bdd3-e5162c8a7a39', 4, 'Good washing machine, but a bit noisy.');

-- Insert data into inventory
INSERT INTO inventory (id, product_id, quantity)
VALUES
    ('e9142592-1be2-4f9c-a34c-17f9c5ff3009', 'bc7e29be-8fd2-4e7f-87a4-4c7c3a90a53f', 100),
    ('3b5ad61d-bc2c-4060-b6a0-c24b15602c95', '7b9e8c92-0971-47c4-bdd3-e5162c8a7a39', 50);
