# Browser agent security policy

Browser control requires a per-origin user grant. Final financial actions are
always handed to the user: the agent never activates a control whose name,
action URL path, or enclosing form indicates buying, payment, placing or
confirming an order, donation, subscription, authorization, finalization,
transfer, or tipping. The handover asks the user to review and press the control
themselves and displays inspected form values with sensitive values masked.
Returning from handover requires fresh observation; it never retries or
authorizes the financial action. This applies regardless of DOM field presence
or values, including transactions whose amount and recipient live in JavaScript.

Cart and quantity additions may run unattended only when the label has no
financial commit intent and the enclosing form does not indicate purchase,
checkout, payment, order submission, or payment fields. Destructive actions on
buttons, links, menu items, options and switches require approval, as do other
consequential and submit classes. Recognized sensitive pages and fields require
human handover.

## Accepted residual risk and known limitations

Labels and accessible names are page-controlled. A malicious site that the user
has granted access can mislabel a consequential control as a benign action.
This policy accepts that residual risk; classification is not a trusted contract
for arbitrary page JavaScript. Mitigations are per-origin grants, never allowing
unattended recognized financial commits, approval for destructive and submit
classes, and sensitive-page handover.

Sensitive observation protection is partially closed: visible recognized fields
in frames and open shadow roots are checked, but closed shadow roots,
unrecognized fields and sensitive rendered text may remain undetected.
Unresolved or closed-shadow keyboard focus is blocked. Listed password-manager
vault domains are blocked, but self-hosted or unlisted unlocked vaults without
recognized sensitive inputs may remain accessible. These are known limitations,
not claims of complete sensitive-content or vault detection.
