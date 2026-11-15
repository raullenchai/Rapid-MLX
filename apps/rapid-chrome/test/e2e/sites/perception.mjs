// Repeated-card and split subtotal shapes reconstructed from the r3 grocery runs.
export function perceptionPage(products, cart) {
  const money = n => `$${n.toFixed(2)}`;
  return `<a href="/perception/cart">View cart</a><div role="grid"><div role="row">${products.slice(0,3).map(p => `<div role="gridcell"><h2>${p.name}</h2><a href="/perception/cart">${p.name}</a><span>${money(p.price)}</span><form method="post" action="/api/cart"><input type="hidden" name="id" value="${p.id}"><input type="hidden" name="quantity" value="1"><button type="submit">Add 1 to order</button></form></div>`).join('')}</div></div><div role="row"><span>Subtotal</span><span>${money(Object.entries(cart).reduce((sum,[id,n]) => sum+products.find(p => p.id===id).price*n,0))}</span></div>`;
}
