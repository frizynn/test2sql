-- Sales data from a bakery/alfajor shop (Nivii).
-- Single table with 24,212 rows covering Sep 22 – Nov 19, 2024.
-- All columns are NOT NULL.

CREATE TABLE sales (
    -- Date of the sale in MM/DD/YYYY text format (no zero-padding).
    -- Examples: '10/4/2024', '11/13/2024', '9/22/2024'
    -- Range: 9/22/2024 to 11/19/2024
    -- WARNING: This is a TEXT column. Do NOT use date functions like month() or year().
    -- Use LIKE patterns for filtering: date LIKE '10/%' for October, date LIKE '11/%' for November.
    date TEXT NOT NULL,

    -- Day of the week in ENGLISH.
    -- Valid values: Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
    -- Spanish→English mapping for user queries:
    --   lunes→Monday, martes→Tuesday, miércoles→Wednesday, jueves→Thursday,
    --   viernes→Friday, sábado→Saturday, domingo→Sunday
    week_day TEXT NOT NULL,

    -- Time of the sale in HH:MM 24-hour format.
    -- Examples: '10:00', '16:55', '21:45'
    -- Range: 10:00 to 21:59
    hour TEXT NOT NULL,

    -- Ticket/receipt identifier with prefix and sequential number.
    -- Prefixes: FCB (regular sales, ~24,052 rows), FCA (bulk/wholesale, ~116 rows),
    --           NCB (credit notes/returns, ~43 rows), NCA (credit notes, ~1 row)
    -- Format: 'FCB 0003-000024735', 'NCA 0003-000000003'
    ticket_number TEXT NOT NULL,

    -- Waiter/salesperson ID.
    -- Valid values: '0', '51', '52', '101', '102', '103', '104', '105', '116'
    -- (9 distinct waiters; stored as text)
    waiter TEXT NOT NULL,

    -- Product name (68 distinct products). Mixed abbreviations and Spanish.
    -- Top 15 products by frequency:
    --   'Alf. 150 aniv. Suelto', 'Alfajor Sin Azucar Suelto', 'Alfajor choc x un',
    --   'Alfajor choc blanco nuez x un', 'Alfajor merengue x un',
    --   'Alfajor mixto caja x12un', 'Alfajor 70 cacao x un', 'Alfajor Super DDL x un',
    --   'Alfajor mixto caja x6un', 'Alf. 150 aniv. X 8 unidades',
    --   'Alfajor choc caja x12un', 'Conito choc caja x6un',
    --   'Conito coco y ddl suelto', 'Alfajor Sin Azucar x9 Un',
    --   'Alfajor choc caja x6un'
    -- Use LIKE '%keyword%' for fuzzy matching when the exact name is unknown.
    product_name TEXT NOT NULL,

    -- Quantity sold. Usually a positive integer, but stored as REAL.
    -- Negative values (e.g., -1.0, -4.0) indicate returns/credit notes.
    -- Fractional values exist (e.g., 0.5).
    -- Range: -4.0 to 30.0
    quantity REAL NOT NULL,

    -- Unit price of the product.
    -- 0.0 indicates a promotional/free item (~2,959 rows).
    -- Range: 0.0 to 75,000.0
    unitary_price REAL NOT NULL,

    -- Line total = quantity × unitary_price.
    -- Can be negative for returns.
    -- Range: -43,000.0 to 605,000.0
    total REAL NOT NULL
);
