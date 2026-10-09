// A deterministic catalog of ~330 products. The task's targets sit among
// near-duplicates (other egg counts, organic and conventional avocados,
// several milks) so the right one has to be read, not guessed.
const CATALOG = (() => {
  const items = [];
  let n = 1000;
  const add = (name, size, price, extra = {}) => items.push({ sku: String(n++), name, size, price, stock: true, ...extra });

  add("Cage-Free Large White Eggs", "18 ct", 5.49);
  add("Cage-Free Large Brown Eggs", "24 ct", 7.29);
  add("Organic Large Brown Eggs", "18 ct", 7.49);
  add("Pasture-Raised Large Eggs", "12 ct", 6.99);
  add("Large White Eggs", "60 ct", 12.99);
  add("Liquid Egg Whites", "6 × 16 oz", 11.49);
  add("Organic Liquid Eggs", "4 × 32 oz", 13.99);
  add("Hard-Boiled Eggs, Peeled", "24 ct", 8.79);
  add("Egg Bites, Bacon & Gruyère", "10 ct", 12.49);
  add("Free-Range Large Brown Eggs", "24 ct", 7.99);
  add("Omega-3 Large Eggs", "18 ct", 6.49);
  add("Organic Large Brown Eggs", "24 ct", 8.99, { target: "eggs" });
  add("Organic Pasture-Raised Eggs", "24 ct", 10.99);
  add("Duck Eggs", "12 ct", 9.99, { stock: false });

  add("Bananas", "3 lb", 1.99, { target: "bananas" });
  add("Organic Bananas", "3 lb", 2.79);
  add("Baby Bananas", "2 lb", 3.49);
  add("Plantains", "4 ct", 2.99);
  add("Banana Chips", "40 oz", 8.99);

  add("Whole Milk", "1 gal × 2", 6.29, { stock: false, target: "milk_whole" });
  add("2% Reduced Fat Milk", "1 gal × 2", 6.19, { target: "milk_2" });
  add("1% Lowfat Milk", "1 gal × 2", 6.09);
  add("Organic Whole Milk", "half gal × 3", 11.99);
  add("Lactose-Free Whole Milk", "half gal × 2", 8.49);
  add("Fat Free Milk", "1 gal × 2", 5.99);
  add("Chocolate Milk", "half gal", 3.79);
  add("Oat Milk, Original", "64 oz × 3", 10.99);
  add("Almond Milk, Unsweetened", "half gal × 3", 8.99);
  add("Whole Milk Greek Yogurt, Plain", "48 oz", 7.49);

  add("Hass Avocados", "6 ct", 6.99, { target: "avocados" });
  add("Organic Hass Avocados", "5 ct", 7.99);
  add("Avocado Oil", "2 L", 13.99);
  add("Guacamole Cups", "20 × 2 oz", 12.49);
  add("Avocado Toast Seasoning", "12 oz", 4.99);

  const fill = {
    Produce: ["Strawberries 2 lb", "Blueberries 18 oz", "Raspberries 12 oz", "Gala Apples 5 lb", "Honeycrisp Apples 4 lb", "Navel Oranges 8 lb", "Clementines 5 lb", "Red Seedless Grapes 4 lb", "Baby Spinach 1 lb", "Spring Mix 1 lb", "Romaine Hearts 6 ct", "Broccoli Florets 3 lb", "Mini Sweet Peppers 2 lb", "Russet Potatoes 15 lb", "Yellow Onions 10 lb", "Garlic 2 lb", "Carrots 10 lb", "Celery 2 ct", "Cucumbers 3 ct", "Roma Tomatoes 4 lb", "Lemons 5 lb", "Limes 3 lb", "Mangoes 6 ct", "Pineapple 1 ct", "Watermelon 1 ct", "Sweet Potatoes 5 lb", "Mushrooms 24 oz", "Asparagus 2 lb", "Green Beans 2 lb", "Kale 2 lb"],
    Dairy: ["Butter, Salted 4 lb", "Butter, Unsalted 4 lb", "Shredded Mozzarella 5 lb", "Sharp Cheddar 2 lb", "Parmigiano Reggiano 1.5 lb", "Cream Cheese 6 × 8 oz", "Sour Cream 3 lb", "Cottage Cheese 3 lb", "Heavy Whipping Cream half gal", "Half and Half 2 × qt", "String Cheese 48 ct", "Brie 2 × 8 oz"],
    Bakery: ["Sourdough Loaf 2 ct", "Croissants 12 ct", "Bagels 12 ct", "Blueberry Muffins 12 ct", "Dinner Rolls 36 ct", "Whole Wheat Bread 2 ct", "Tortillas 30 ct", "Pita Bread 16 ct", "Chocolate Chip Cookies 24 ct", "Cinnamon Rolls 12 ct"],
    Pantry: ["Extra Virgin Olive Oil 2 L", "Basmati Rice 25 lb", "Jasmine Rice 25 lb", "Spaghetti 8 × 1 lb", "Marinara Sauce 3 × 32 oz", "Peanut Butter 2 × 40 oz", "Almond Butter 27 oz", "Honey 3 lb", "Maple Syrup 33.8 oz", "Rolled Oats 10 lb", "Granola 35 oz", "Black Beans 8 × 15 oz", "Chickpeas 8 × 15 oz", "Chicken Broth 6 × 32 oz", "Coconut Milk 6 × 13.5 oz", "Canned Tuna 8 × 7 oz", "Quinoa 4.5 lb", "Flour 25 lb", "Sugar 10 lb", "Sea Salt 2.2 lb", "Black Peppercorns 12.7 oz", "Ground Coffee 3 lb", "Whole Bean Coffee 2.5 lb", "Green Tea 100 ct", "Mixed Nuts 2.5 lb", "Cashews 2.5 lb", "Dried Mango 40 oz", "Tortilla Chips 3 lb", "Salsa 2 × 38 oz", "Pretzels 3.5 lb"],
    Meat: ["Chicken Breasts 6 lb", "Chicken Thighs 5 lb", "Ground Beef 88% Lean 4 lb", "Ribeye Steaks 3 lb", "Pork Tenderloin 4 lb", "Bacon 4 × 1 lb", "Atlantic Salmon 3 lb", "Shrimp, Raw 2 lb", "Rotisserie Chicken 1 ct", "Turkey Breast Deli 2 lb", "Italian Sausage 3 lb", "Lamb Chops 2 lb"],
    Frozen: ["Frozen Blueberries 4 lb", "Frozen Mixed Vegetables 5.5 lb", "Frozen Broccoli 4 lb", "Chicken Potstickers 4.5 lb", "Cheese Pizza 4 ct", "Ice Cream, Vanilla 2 × 1.5 qt", "Frozen Waffles 60 ct", "Edamame 5 lb", "Frozen Shrimp 2 lb", "Mango Chunks 5 lb"],
    Household: ["Paper Towels 12 rolls", "Bath Tissue 30 rolls", "Dish Soap 2 × 90 oz", "Laundry Detergent 146 oz", "Dishwasher Pods 115 ct", "Trash Bags 200 ct", "Aluminum Foil 500 sq ft", "Zipper Bags 4 × 75 ct", "Disinfecting Wipes 5 × 85 ct", "Hand Soap 3 × 1 L", "Facial Tissue 12 boxes", "Sponges 24 ct"],
  };
  const brands = ["Harbor Select", "Green Acre", "Signature", "Daily Table"];
  let seed = 7;
  const rand = () => ((seed = (seed * 9301 + 49297) % 233280) / 233280);
  for (const [dept, names] of Object.entries(fill)) {
    for (const raw of names) {
      for (const brand of brands.slice(0, dept === "Produce" ? 1 : 2 + Math.floor(rand() * 2))) {
        const m = raw.match(/^(.*?) ((?:\d|half )\S* ?.*)$/);
        const [name, size] = m ? [m[1], m[2]] : [raw, ""];
        add(`${brand} ${name}`, size, Math.round((3 + rand() * 30) * 100) / 100 - 0.01, { stock: rand() > 0.08, dept });
      }
    }
  }
  return items;
})();
